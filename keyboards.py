from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, KeyboardButton
inline_btn_1 = InlineKeyboardButton('Описание', callback_data='button1')
#inline_kb1 = InlineKeyboardMarkup().add(inline_btn_1)
inline_btn_2 = InlineKeyboardButton('Обработка NST', callback_data='button2')
inline_kb_full = InlineKeyboardMarkup(row_width=3)
#inline_kb_full.add(InlineKeyboardButton('Вторая кнопка', callback_data='btn2'))
inline_btn_3 = InlineKeyboardButton('Обработка cycleGAN', callback_data='button3')
inline_btn_4 = InlineKeyboardButton('Назад', callback_data='button4')
inline_btn_5 = InlineKeyboardButton('horse2zebra', callback_data='button5')
inline_btn_6 = InlineKeyboardButton('winter2summer', callback_data='button6')
style_gan_kb = InlineKeyboardMarkup(row_width=3)
style_gan_kb.add(inline_btn_5, inline_btn_6,inline_btn_4)
inline_kb_full.add(inline_btn_1,inline_btn_2, inline_btn_3)
small_inline_kb = InlineKeyboardMarkup(row_width=1)
small_inline_kb.add(inline_btn_4)