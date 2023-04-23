import logging
import itertools
import datetime
import cv2
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

from realesrgan import RealESRGANer

import telegram
from telegram import constants, BotCommandScopeAllGroupChats
from telegram import Message, MessageEntity, Update, \
    BotCommand, ChatMember, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, \
    filters, InlineQueryHandler, Application, CallbackContext, CallbackQueryHandler

import replicate


def message_text(message: Message) -> str:
    """
    Returns the text of a message, excluding any bot commands.
    """
    message_text = message.text
    if message_text is None:
        return ''

    for _, text in sorted(message.parse_entities([MessageEntity.BOT_COMMAND]).items(), key=(lambda item: item[0].offset)):
        message_text = message_text.replace(text, '').strip()

    return message_text if len(message_text) > 0 else ''


def high_res(input_path):
    model_name = 'RealESRGAN_x4plus'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    model_path = os.path.join('weights', model_name + '.pth')
    dni_weight = None
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=None)
    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=4,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler)
    logging.info(f'input_path: {input_path}')
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    output_path = input_path.replace(".jpg", "_high.jpg")
    cv2.imwrite(output_path, output)
    return output_path


def replicate_high_res(url_path):
    output = replicate.run(
        "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b",
        input={"image": url_path},
    )
    logging.info(output)
    return output


class RSBot:
    """
    Class representing a ChatGPT Telegram Bot.
    """

    def __init__(self, config: dict):
        """
        Initializes the bot with the given configuration and GPT bot object.
        :param config: A dictionary containing the bot configuration
        :param api: WebUIApiHelper object
        """
        self.config = config
        self.commands = [
            BotCommand(command='help', description='Show help message'),
            BotCommand(command='draw', description='draw a picture'),
            BotCommand(command='model', description='change models'),
            BotCommand(command='dress', description='change clothes'),
        ]
        self.group_commands = [
            BotCommand(command='chat', description='Chat with the bot!')
        ] + self.commands
        self.disallowed_message = "Sorry, you are not allowed to use this bot. You can connect to @aipicfree"
        self.budget_limit_message = "Sorry, you have reached your monthly usage limit."
        self.usage = {}
        self.last_message = {}

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Shows the help menu.
        """
        commands = self.group_commands if self.is_group_chat(update) else self.commands
        commands_description = [f'/{command.command} - {command.description}' for command in commands]
        help_text = 'I\'m a pic high resolution bot, talk to me!' + \
                    '\n\n' + \
                    '\n'.join(commands_description) + \
                    '\n\n' + \
                    'Send me a image and I\'ll transcribe it for you!'
        await update.message.reply_text(help_text, disable_web_page_preview=True)

    async def draw(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.is_allowed(update, context):
            logging.warning(f'User {update.message.from_user.name}: {update.message.from_user.id} is not allowed to use this bot')
            await self.send_disallowed_message(update, context)
            return
        await update.message.reply_text("draw")

    async def high(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.is_allowed(update, context):
            logging.warning(f'User {update.message.from_user.name}: {update.message.from_user.id} is not allowed to use this bot')
            await self.send_disallowed_message(update, context)
            return
        message = update.message
        bot = context.bot
        if message.photo:
            # pic_path = await self.down_image(bot, message)
            # high_path = high_res(pic_path)
            # torch.cuda.empty_cache()
            # await message.reply_document(high_path)
            url = await self.get_file_url(bot, message)
            output = replicate_high_res(url)
            await message.reply_document(output)

    async def get_file_url(self, bot, message):
        logging.info("Message contains one photo.")
        file = await bot.getFile(message.photo[-1].file_id)
        logging.info(file)
        file_url = file.file_path
        logging.info(file_url)
        return file_url

    async def down_image(self, bot, message):
        logging.info("Message contains one photo.")
        date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        path = f'download/photo_{date}.jpg'
        logging.info(f"{path}")
        file = await bot.getFile(message.photo[-1].file_id)
        logging.info(file)
        photo_path = await file.download_to_drive(custom_path=path)
        logging.info(photo_path)

        return str(photo_path)

    async def send_disallowed_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Sends the disallowed message to the user.
        """
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=self.disallowed_message,
            disable_web_page_preview=True
        )

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handles errors in the telegram-python-bot library.
        """
        logging.error(f'Exception while handling an update: {context.error}')

    def is_group_chat(self, update: Update) -> bool:
        """
        Checks if the message was sent from a group chat
        """
        return update.effective_chat.type in [
            constants.ChatType.GROUP,
            constants.ChatType.SUPERGROUP
        ]

    async def is_user_in_group(self, update: Update, context: CallbackContext, user_id: int) -> bool:
        """
        Checks if user_id is a member of the group
        """
        try:
            chat_member = await context.bot.get_chat_member(update.message.chat_id, user_id)
            return chat_member.status in [ChatMember.OWNER, ChatMember.ADMINISTRATOR, ChatMember.MEMBER]
        except telegram.error.BadRequest as e:
            if str(e) == "User not found":
                return False
            else:
                raise e
        except Exception as e:
            raise e

    async def is_allowed(self, update: Update, context: CallbackContext) -> bool:
        """
        Checks if the user is allowed to use the bot.
        """
        if self.config['allowed_user_ids'] == '*':
            return True
        
        if self.is_admin(update):
            return True
        
        allowed_user_ids = self.config['allowed_user_ids'].split(',')
        # Check if user is allowed
        if str(update.message.from_user.id) in allowed_user_ids:
            return True

        # Check if it's a group a chat with at least one authorized member
        if self.is_group_chat(update):
            admin_user_ids = self.config['admin_user_ids'].split(',')
            for user in itertools.chain(allowed_user_ids, admin_user_ids):
                if not user.strip():
                    continue
                if await self.is_user_in_group(update, context, user):
                    logging.info(f'{user} is a member. Allowing group chat message...')
                    return True
            logging.info(f'Group chat messages from user {update.message.from_user.name} '
                f'(id: {update.message.from_user.id}) are not allowed')

        logging.info(f'user_name: {update.message.from_user.name}, user_id: {update.message.from_user.id}  is not allowed!!')
        return False

    def is_admin(self, update: Update) -> bool:
        """
        Checks if the user is the admin of the bot.
        The first user in the user list is the admin.
        """
        if self.config['admin_user_ids'] == '-':
            logging.info('No admin user defined.')
            return False

        admin_user_ids = self.config['admin_user_ids'].split(',')

        # Check if user is in the admin user list
        if str(update.message.from_user.id) in admin_user_ids:
            return True

        return False

    def get_reply_to_message_id(self, update: Update):
        """
        Returns the message id of the message to reply to
        :param update: Telegram update object
        :return: Message id of the message to reply to, or None if quoting is disabled
        """
        if self.config['enable_quoting'] or self.is_group_chat(update):
            return update.message.message_id
        return None

    async def post_init(self, application: Application) -> None:
        """
        Post initialization hook for the bot.
        """
        await application.bot.set_my_commands(self.group_commands, scope=BotCommandScopeAllGroupChats())
        await application.bot.set_my_commands(self.commands)

    def run(self):
        """
        Runs the bot indefinitely until the user presses Ctrl+C
        """
        application = ApplicationBuilder() \
            .token(self.config['token']) \
            .proxy_url(self.config['proxy']) \
            .get_updates_proxy_url(self.config['proxy']) \
            .post_init(self.post_init) \
            .concurrent_updates(True) \
            .build()

        application.add_handler(CommandHandler('draw', self.draw))
        application.add_handler(CommandHandler('help', self.help))
        application.add_handler(CommandHandler('start', self.help))

        application.add_handler(MessageHandler(filters.PHOTO, self.high))

        #application.add_error_handler(self.error_handler)

        application.run_polling()
