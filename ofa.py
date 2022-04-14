import os
import sys
import time
import telebot
import torch
import shortuuid
import numpy as np
from fairseq import utils,tasks
from utils import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel
from PIL import Image
from googletrans import Translator

if len(sys.argv)>1:
    token = sys.argv[1]
else:
    token = os.getenv('TG_TOKEN')
    if not token: 
    	print('bot token needed...')
    	quit()

if len(sys.argv)>2:
    my_chat_id = int(sys.argv[2])
else:
    my_chat_id = os.getenv('MY_CHAT')


# assign directory
new_directory = 'images/new'
save_directory = 'images/save'
res_filename = 'images/results.csv'

# Register caption task
tasks.register_task('caption',CaptionTask)

# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False

# Load pretrained ckpt & config
overrides={"bpe_dir":"utils/BPE", "eval_cider":False, "beam":5, "max_len_b":16, "no_repeat_ngram_size":3, "seed":7}
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths('checkpoints/caption.pt'),
        arg_overrides=overrides
    )

# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

# Image transform
from torchvision import transforms
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()
def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

# Construct input for caption task
def construct_sample(image: Image):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample
  
# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

translator = Translator()
#translation = translator.translate("Guten Morgen")
#print translation

bot = telebot.TeleBot(token)

print(f'Main cicle whith cuda is {use_cuda} start...')

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
	bot.reply_to(message, "Бот формирует текстовое описание того, что изображено на картинке. Просто отправь ему изображение...")

@bot.message_handler(content_types='text')
def message_reply(message):
    translation = translator.translate(message.text, dest='ru').text
    bot.reply_to(message, translation)

@bot.message_handler(content_types=['photo'])
def get_photo(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    #image = bot.download_file(file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)
    src = 'images/new/' + message.photo[1].file_id
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    image = Image.open(src) 
    # Construct input sample & preprocess for GPU if cuda available
    sample = construct_sample(image)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    # Run eval step for caption
    with torch.no_grad():
        result, scores = eval_step(task, generator, models, sample)

    #display(image)
    description = result[0]['caption']
    translation = translator.translate(description, dest='ru').text
    file_n = shortuuid.uuid()+'.jpg'
    with open(res_filename, "a") as csvfile:
        csvfile.write(f'{file_n}, {description}\n')
    #bot.reply_to(message, f'На изображении <b>{translation}</b> (<i>{description}</i>).', parse_mode="HTML")
    bot.delete_message(message.chat.id, message.message_id)
    bot.send_photo(message.chat.id, downloaded_file, 
        caption=f'На изображении <b>{translation}</b> (<i>{description}</i>).', 
        parse_mode="HTML")
    if my_chat_id and not message.chat.id == int(my_chat_id):
        us_name = (message.from_user.first_name + 
            ((' '+message.from_user.last_name) if message.from_user.last_name else '') +
            ((' aka '+ message.from_user.username) if message.from_user.username else ''))
        bot.send_photo(int(my_chat_id), downloaded_file,
            caption=f'На изображении <b>{translation}</b> (<i>{description}</i>). От {us_name}', 
            parse_mode="HTML")
    print(description)
    T = transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC)
    img = T(image)
    fs = os.path.join(save_directory, file_n)
    img.save(fs)
    os.remove(src)

#bot.polling(interval=3, timeout=45)
bot.infinity_polling()
