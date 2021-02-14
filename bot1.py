import os
import shutil
from aiogram import Bot, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.dispatcher.filters import state
from aiogram.utils import executor
from nst import image_loader, StyleTransfer, cnn
import io
import torchvision.transforms as transforms
from aiogram.dispatcher.filters.state import State, StatesGroup
import logging
from config import TOKEN
import keyboards as kb
logging.basicConfig(level=logging.INFO)
bot = Bot(token=TOKEN)

storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


class Form(StatesGroup):
    st_nst = State()
    st_gan = State()


@dp.message_handler(commands=['start'], state='*')
async def process_start_command(message: types.Message, state : FSMContext):
    await message.reply("Привет!\nВыбери, что тебе нужно:", reply_markup=kb.inline_kb_full)
    await state.finish()

dict_photo = {}


@dp.callback_query_handler(lambda c: c.data and c.data.startswith('button'), state='*')
async def inline_kb_answer_callback_handler(query: types.CallbackQuery, state: FSMContext):
    await query.answer()
    await state.finish()
    code = query.data[-1]
    if code.isdigit():
        code = int(code)

    if code == 1:
        await bot.send_message(query.from_user.id, text='Этот бот позволяет применить несколько стилей к любому изображению двумя методами:\n1. С использованием NST - наложением нескольких стилей на изображение. \n2. С использованием предобученного cycleGAN. ', reply_markup = kb.small_inline_kb)
    elif code == 2:
        await Form.st_nst.set()
        await bot.send_message(query.from_user.id, text='Теперь загрузи три фотографии в следующем порядке:\n1. Изображение контента\n2.Изображения стиля №1\n3.Изображение стиля №2', reply_markup = kb.small_inline_kb)
    elif code == 3:
        await bot.send_message(query.from_user.id, text='Теперь выбери предобученный GAN или можешь вернуться назад:', reply_markup = kb.style_gan_kb)


    elif code == 4:
        await bot.send_message(query.from_user.id,text = 'Главное меню', reply_markup = kb.inline_kb_full)

    elif code == 5:
        await Form.st_gan.set()
        async with state.proxy() as data:  # запись
            data['gan_style'] = 'horse2zebra'
        await bot.send_message(query.from_user.id, text='Теперь загрузи фотографию.', reply_markup=kb.small_inline_kb)

    elif code == 6:
        await Form.st_gan.set()
        async with state.proxy() as data:  # запись
            data['gan_style'] = 'winter2summer_yosemite_pretrained'
        await bot.send_message(query.from_user.id, text='Теперь загрузи фотографию.', reply_markup=kb.small_inline_kb)

@dp.message_handler(content_types=['photo'], state=Form.st_nst)
async def handle_docs_photo(message: types.Message, state : FSMContext):
    photo_file_id = message.photo[-1].file_id
    user_id = message.from_user.id
    if user_id in dict_photo:
        dict_photo[user_id].append(photo_file_id)
    else:
        dict_photo[user_id] = [photo_file_id]

    path_user_photos = os.getcwd() + os.sep + "User_photos"
    if len(dict_photo[user_id]) == 3:
        state.finish()
        try:
            os.mkdir(os.path.join(path_user_photos, str(user_id)))
        except OSError:
            pass

        for i, file_id in enumerate(dict_photo[user_id]):
            await bot.download_file_by_id(file_id, os.path.join(path_user_photos, str(user_id), str(i) + '.jpg'))
        path_user_photos = os.getcwd() + os.sep + "User_photos"
        file_names = ['0.jpg', '1.jpg', '2.jpg']
        style_img1 = image_loader(os.path.join(path_user_photos, str(user_id), str(file_names[0])))
        content_img = image_loader(os.path.join(path_user_photos, str(user_id), str(file_names[2])))
        style_img2 = image_loader(os.path.join(path_user_photos, str(user_id), str(file_names[1])))

        result = await StyleTransfer(style_img1, style_img2, content_img, cnn).run_style_transfer()
        image = result.cpu()
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        await bot.send_photo(message.chat.id, photo=byte_im)
        dict_photo[user_id] = []
        await bot.send_message(message.from_user.id, 'Теперь можно вернуться назад.', reply_markup=kb.small_inline_kb)


# CycleGAN обработка
@dp.message_handler(content_types=['photo'], state=Form.st_gan)
async def handle_CGAN_photo(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        gan = data['gan_style']

    async with state.proxy() as data:
        gan = data.get('gan_style', 'horse2zebra')

    path_cgan_files = os.getcwd() + os.sep + "GAN_files"
    path_cgan_results = os.getcwd() + os.sep + "GAN_results"

    try:
        user_id = message.from_user.id
        input_dir = os.path.join(path_cgan_files, str(user_id))
        fake_path = os.path.join(path_cgan_results, "images", f"{str(user_id)}_fake.png")

        if os.path.isdir(input_dir): shutil.rmtree(input_dir)
        os.mkdir(input_dir)

        await message.photo[-1].download( os.path.join(input_dir, f"{str(user_id)}.jpg"))

        input_dir_for_bash = f'"{input_dir}"'
        output_dir_for_bash = f'"{path_cgan_results}"'

        os.system(f"cd pytorch-CycleGAN-and-pix2pix && "  
                  f"python test.py --dataroot {input_dir_for_bash} "
                  f"--name {gan} "
                  f"--results_dir {output_dir_for_bash} "
                  f"--preprocess scale_width "
                  f"--model test --gpu_ids -1 --no_dropout")

        with open (fake_path, 'rb') as file:
            await message.answer_photo(photo=file)
            await message.answer('Обработка закончена.', reply_markup=kb.small_inline_kb)

    except BaseException as e:
        logging.info(f'Ошибка\n {e}')
    finally:
        pass

@dp.message_handler(content_types=types.ContentTypes.ANY, state='*')
async def echo_message1(msg: types.Message, state: FSMContext):
    await msg.answer("Команда не понята.\nЧтобы начать сначала, жми\n/start")

if __name__ == '__main__':
    executor.start_polling(dp)