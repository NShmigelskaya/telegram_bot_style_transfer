# telegram_bot_style_transfer - @change_style_bot
This is a telegram bot that processes images using NST and CycleGAN. The bot is written in the aiogram library. 
The bot1.py file contains the states and some handlers: a handler to start the bot using the /start command; a handler that accepts a callback when pressing buttons embedded in the inline keyboard (the buttons are written in the keyboards.py file); handle_docs_photo handler captures 3 photos for NST processing; handle_CGAN_photo is a handler for processing a photo using a pre-trained CycleGAN, there are 2 CycleGANs for the user to choose: horse2zebra, winter2summer.
The NST code is written in the nst.py file, where the StyleTransfer class has been added, into which the __init__, change_layers, get_style_model_and_losses, get_input_optimizer, run_style_transfer functions are placed. 
Also, in general, the bot is asynchronous.

