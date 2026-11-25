"""Веб-сайт для распознавания каталога одежды через камеру в браузере.
Запуск: py -m streamlit run web_catalog.py
"""
from datetime import datetime
from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from torchvision import models

# Настройка страницы сайта
st.set_page_config(page_title='Каталог одежды — Распознавание', page_icon='👕', layout='centered')

CATALOG_MAP = {
    't-shirt': 'Футболка / топ',
    'jersey': 'Футболка / спортивная форма',
    'sweatshirt': 'Толстовка / свитшот',
    'cardigan': 'Кардиган / кофта',
    'wool': 'Свитер / трикотаж',
    'jean': 'Джинсы',
    'trouser': 'Брюки / штаны',
    'miniskirt': 'Юбка',
    'overskirt': 'Юбка / накидка',
    'gown': 'Платье',
    'abaya': 'Платье / накидка',
    'coat': 'Пальто',
    'trench coat': 'Тренч / плащ',
    'poncho': 'Пончо',
    'suit': 'Костюм / пиджак',
    'fur coat': 'Шуба / мех',
    'backpack': 'Рюкзак',
    'handbag': 'Сумка',
    'purse': 'Сумочка',
    'running shoe': 'Кроссовки',
    'loafer': 'Туфли / лоферы',
    'clog': 'Сабо / мюли',
    'cowboy boot': 'Ботинки / сапоги',
    'sunglasses': 'Солнцезащитные очки',
    'hat': 'Головной убор',
    'cap': 'Кепка / бейсболка',
    'sock': 'Носки',
    'mitten': 'Перчатки / варежки',
}


def classify_clothing(raw_label: str) -> str:
    low = raw_label.lower()
    for key, ru_name in CATALOG_MAP.items():
        if key in low:
            return ru_name
    return f'Другое ({raw_label})'


@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights).to(device)
    model.eval()
    return model, weights.transforms(), weights.meta['categories'], device


model, preprocess, categories, device = load_model()

# Интерфейс сайта
st.title('👕 Распознавание каталога одежды')
st.write('Сделайте снимок вещи на камеру или загрузите фото, чтобы определить её категорию.')

if 'history' not in st.session_state:
    st.session_state.history = []

tab_cam, tab_file = st.tabs(['📷 Камера', '📁 Загрузить фото'])

with tab_cam:
    photo = st.camera_input('Наведите камеру на вещь и нажмите Take Photo')

with tab_file:
    uploaded = st.file_uploader('Выберите изображение одежды', type=['jpg', 'jpeg', 'png', 'webp'])

image_source = photo if photo is not None else uploaded

if image_source is not None:
    pil_img = Image.open(image_source).convert('RGB')

    # Обрезаем центральный квадрат, чтобы меньше мешал фон по краям
    w, h = pil_img.size
    side = int(min(w, h) * 0.75)
    left, top = (w - side) // 2, (h - side) // 2
    cropped = pil_img.crop((left, top, left + side, top + side))

    inp = preprocess(cropped).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(inp), dim=1)[0]
        top3_prob, top3_idx = torch.topk(probs, k=3)

    best_conf = float(top3_prob[0].item())
    best_raw = categories[int(top3_idx[0].item())]
    best_cat = classify_clothing(best_raw)

    st.subheader(f'Результат: {best_cat}')
    st.metric(label='Уверенность модели', value=f'{best_conf:.1%}')

    # Топ-3 варианта на случай похожих фасонов
    with st.expander('Показать топ-3 гипотезы модели'):
        for p, idx in zip(top3_prob, top3_idx):
            raw = categories[int(idx.item())]
            st.write(f'- **{classify_clothing(raw)}** (`{raw}`) — {float(p.item()):.1%}')

    if st.button('💾 Сохранить вещь в каталог'):
        out_dir = Path('results')
        out_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'web_item_{stamp}.jpg'
        pil_img.save(out_dir / fname)

        record = {
            'Время': datetime.now().strftime('%H:%M:%S'),
            'Категория': best_cat,
            'Уверенность': f'{best_conf:.1%}',
            'Файл': fname,
        }
        st.session_state.history.append(record)
        pd.DataFrame(st.session_state.history).to_csv(
            out_dir / 'web_catalog.csv', index=False, encoding='utf-8-sig'
        )
        st.success(f'Вещь добавлена в каталог (`results/{fname}`)!')

if st.session_state.history:
    st.divider()
    st.subheader('📋 Распознанные вещи за сессию')
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)