## Интерфейс в виде телеграм-бота к инференсу OFA (https://github.com/OFA-Sys/OFA)

### Пошаговая инструкция, как запустить у себя. В конце есть еще вариат с docker. 

Все команды, написаные ниже, нужно вводить в терминал. Предполагается, что это терминал юниксовый, хотя за редким исключением (типа wget) будет работать и под виндой.

Для начала нужно склонировать два репозитория: самой модели и библиотеки fairseq

```bash
git clone https://github.com/pytorch/fairseq.git
git clone https://github.com/OFA-Sys/OFA.git
```

После этого нужно закинуть в репозиторий модели веса, которые хранятся отдельно.


```bash
mkdir -p OFA/checkpoints/
wget https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/caption_large_best_clean.pt
mv caption_large_best_clean.pt OFA/checkpoints/caption.pt
```

Исходим из того, что conda установлена... Вообще все можно и без конды - она просто позволяет сделать отдельное окружение под конкретный проект, что бы не иметь конфликтов версий установленных библиотек... Вообще без нее можно обойтись.


```bash
conda create --name ofa
conda activate ofa 
```

нужно поставить pytorch, но не обязательно прям именно такую версию и именно так... Я, например, ставил через конду, но тут напишу через pip... вообще можно посмотрет тут: https://pytorch.org/get-started/locally/ Да, что бы работало все на видеокарте в системе должны быть драйвера, с ними может бытьотдельная развлекуха.


```bash
python -m pip install --upgrade pip
pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```
Потом все, что нужно для модели и api сервисов

```bash
pip install -r requirements.txt
pip install -r OFA/requirements.txt
```
Дальше ставим fairseq

```bash
cd fairseq
python -m pip install fairseq --use-feature=in-tree-build ./
```

Скрипт ofa.py кидаем в репозиторий OFA. дальше просто запускаем скрипт, в качестве параметра передаем токен бота (не забыть свой подставить!).

```bash
python ofa.py 123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
```

PS В скрипте еще есть раздел с настройкой структуры каталогов куда картинки предварительно сохраняются и где ведется некая маленькая база результатов: 

```python
# assign directory
new_directory = 'images/new'
save_directory = 'images/save'
res_filename = 'images/results.csv'
```

По умолчанию это все тоже в репозитории OFA... Каталоги и csv лучше заранее создать (хз, не проверял.) 

### UPDATE: docker! Делаем следующее:  

```bash
git clone https://github.com/Nehc/ofa_telegram.git
cd ofa_telegram
docker build --tag="nehcy/ofa" .
docker run --gpus all --name ofa -v "${PWD}/images":/home/OFA/ofa_telegram/images -e TG_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11 
```
так будет сохранять базу картинок в каталог ofa_telegram/images.

\***-gpus all** - предполагает поддержку GPU в докер. Если этого нет - можно без этого параметра. 

@article{wang2022OFA,
  title={Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework},
  author={Wang, Peng and Yang, An and Men, Rui and Lin, Junyang and Bai, Shuai and Li, Zhikang and Ma, Jianxin and Zhou, Chang and Zhou, Jingren and Yang, Hongxia},
  journal={arXiv preprint arXiv:2202.03052},
  year={2022}
}

