import base64
from collections import Counter
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(
    page_title="WARDROBE.AI — Помощник каталога одежды",
    page_icon="🧥",
    layout="wide",
)

FASHION_CLASSES = [
    "Футболка / Топ",     # 0
    "Брюки / Джинсы",     # 1
    "Свитер / Пуловер",   # 2
    "Платье",             # 3
    "Пальто / Куртка",    # 4
    "Сандалия",           # 5
    "Рубашка",            # 6
    "Кроссовок",          # 7
    "Сумка",              # 8
    "Ботинок",            # 9
]

NAMED_COLORS = [
    ("Чёрный", (24, 24, 27)),
    ("Белый", (244, 244, 245)),
    ("Серый", (136, 140, 148)),
    ("Тёмно-синий", (30, 58, 138)),
    ("Голубой", (96, 165, 250)),
    ("Красный", (220, 38, 38)),
    ("Бордовый", (127, 29, 29)),
    ("Розовый", (236, 72, 153)),
    ("Оранжевый", (249, 115, 22)),
    ("Жёлтый", (250, 204, 21)),
    ("Зелёный", (34, 197, 94)),
    ("Коричневый", (120, 72, 42)),
    ("Бежевый", (214, 194, 165)),
    ("Фиолетовый", (139, 92, 246)),
]


def svg_to_uri(svg_text: str) -> str:
    encoded = base64.b64encode(svg_text.strip().encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def build_side_figures_uri() -> str:
    """Создаёт стильных персонажей в трендовой одежде по левому и правому краям."""
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
      <defs>
        <linearGradient id="skyWash" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#ecfccb"/>
          <stop offset="50%" stop-color="#f7fee7"/>
          <stop offset="100%" stop-color="#e0f2fe"/>
        </linearGradient>
      </defs>
      <rect width="1600" height="900" fill="url(#skyWash)"/>

      <!-- Левый арт-персонаж в трендовом оверсайз-бомбере и широких джинсах -->
      <g transform="translate(18, 140)">
        <!-- Голова и причёска -->
        <circle cx="115" cy="95" r="46" fill="#fed7aa" stroke="#1e293b" stroke-width="3"/>
        <path d="M68 86 C70 36 162 36 164 88 C146 66 94 66 68 86 Z" fill="#1e293b"/>
        <!-- Очки в тонкой оправе -->
        <rect x="88" y="84" width="22" height="15" rx="4" fill="#ffffff" fill-opacity="0.75" stroke="#0f172a" stroke-width="2.2"/>
        <rect x="118" y="84" width="22" height="15" rx="4" fill="#ffffff" fill-opacity="0.75" stroke="#0f172a" stroke-width="2.2"/>
        <line x1="110" y1="91" x2="118" y2="91" stroke="#0f172a" stroke-width="2.2"/>
        <!-- Оверсайз-бомбер -->
        <path d="M56 152 L174 152 L214 238 L174 268 L156 226 L156 430 L74 430 L74 226 L34 268 L16 238 Z"
              fill="#334155" stroke="#0f172a" stroke-width="3" stroke-linejoin="round"/>
        <path d="M95 152 L115 210 L135 152" fill="#f8fafc" stroke="#0f172a" stroke-width="2.5"/>
        <line x1="115" y1="210" x2="115" y2="430" stroke="#94a3b8" stroke-width="2.5" stroke-dasharray="6 5"/>
        <!-- Резинка пояса и манжеты -->
        <rect x="74" y="410" width="82" height="22" rx="6" fill="#1e293b"/>
        <!-- Широкие трендовые джинсы -->
        <path d="M76 432 L154 432 L164 700 L118 700 L115 520 L112 700 L66 700 Z"
              fill="#60a5fa" stroke="#1e3a8a" stroke-width="3" stroke-linejoin="round"/>
        <!-- Кроссовки -->
        <path d="M56 700 L118 700 L124 726 L52 726 Z" fill="#ffffff" stroke="#0f172a" stroke-width="2.5"/>
        <path d="M116 700 L172 700 L178 726 L114 726 Z" fill="#ffffff" stroke="#0f172a" stroke-width="2.5"/>
      </g>

      <!-- Правый арт-персонаж в трендовом укороченном жакете, юбке/брюках и сумке -->
      <g transform="translate(1355, 130)">
        <circle cx="115" cy="95" r="45" fill="#fde68a" stroke="#1e293b" stroke-width="3"/>
        <path d="M68 92 C66 34 166 34 166 96 C158 145 162 195 152 230 L134 220 C142 175 140 120 134 84 C112 68 90 70 68 92 Z"
              fill="#451a03"/>
        <!-- Трендовый жакет / кардиган -->
        <path d="M62 152 L168 152 L204 236 L168 262 L152 218 L152 370 L78 370 L78 218 L42 262 L26 236 Z"
              fill="#f43f5e" stroke="#881337" stroke-width="3" stroke-linejoin="round"/>
        <path d="M96 152 L115 202 L134 152" fill="#fff1f2" stroke="#881337" stroke-width="2.4"/>
        <!-- Брюки палаццо -->
        <path d="M78 370 L152 370 L172 690 L122 690 L115 470 L108 690 L58 690 Z"
              fill="#1e293b" stroke="#0f172a" stroke-width="3" stroke-linejoin="round"/>
        <!-- Сумка на плече -->
        <path d="M150 185 C195 190 205 240 195 280" fill="none" stroke="#0f172a" stroke-width="3"/>
        <rect x="172" y="274" width="48" height="42" rx="9" fill="#a3e635" stroke="#365314" stroke-width="2.5"/>
        <!-- Обувь -->
        <path d="M52 690 L110 690 L114 714 L48 714 Z" fill="#fef08a" stroke="#0f172a" stroke-width="2.5"/>
        <path d="M120 690 L178 690 L182 714 L116 714 Z" fill="#fef08a" stroke="#0f172a" stroke-width="2.5"/>
      </g>
    </svg>
    """.strip()
    return svg_to_uri(svg)


def make_showcase_circle_uri(kind: str) -> str:
    """Рисует яркую круглую карточку трендовой одежды для витрины в духе референса."""
    if kind == "bomber":
        inner = """
        <rect width="180" height="180" fill="#fef3c7"/>
        <path d="M52 46 L128 46 L154 78 L132 96 L118 80 L118 144 L62 144 L62 80 L48 96 L26 78 Z"
              fill="#3b82f6" stroke="#1e3a8a" stroke-width="3" stroke-linejoin="round"/>
        <line x1="90" y1="46" x2="90" y2="144" stroke="#dbeafe" stroke-width="2.5"/>
        <rect x="62" y="132" width="56" height="12" rx="4" fill="#1d4ed8"/>
        """
    elif kind == "dress":
        inner = """
        <rect width="180" height="180" fill="#fce7f3"/>
        <path d="M74 38 L106 38 L114 68 L146 144 L34 144 L66 68 Z"
              fill="#ec4899" stroke="#831843" stroke-width="3" stroke-linejoin="round"/>
        <rect x="64" y="66" width="52" height="9" rx="3" fill="#fde047" stroke="#831843" stroke-width="2"/>
        """
    else:
        inner = """
        <rect width="180" height="180" fill="#dcfce7"/>
        <path d="M28 112 L64 112 L94 68 L120 78 L140 102 L162 108 L162 126 L28 126 Z"
              fill="#10b981" stroke="#064e3b" stroke-width="3" stroke-linejoin="round"/>
        <rect x="28" y="122" width="134" height="12" rx="4" fill="#ffffff" stroke="#064e3b" stroke-width="2.5"/>
        """
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="180" height="180" viewBox="0 0 180 180">{inner}</svg>'
    return svg_to_uri(svg)


SIDE_BG_URI = build_side_figures_uri()
CIRCLE_1_URI = make_showcase_circle_uri("bomber")
CIRCLE_2_URI = make_showcase_circle_uri("dress")
CIRCLE_3_URI = make_showcase_circle_uri("sneaker")

st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("{SIDE_BG_URI}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #1e293b;
        }}
        .top-bar {{
            background: linear-gradient(180deg, #a3e635 0%, #84cc16 100%);
            border: 2px solid #65a30d;
            border-radius: 14px;
            padding: 14px 24px;
            margin-bottom: 14px;
            box-shadow: 0 6px 16px rgba(101, 163, 13, 0.22);
            text-align: center;
        }}
        .brand-logo {{
            font-size: 2.2rem;
            font-weight: 900;
            letter-spacing: 0.04em;
            color: #1e293b;
            margin: 0;
            line-height: 1.1;
        }}
        .brand-logo span {{
            color: #15803d;
        }}
        .brand-tagline {{
            margin: 4px 0 0 0;
            font-size: 0.96rem;
            font-weight: 600;
            color: #365314;
        }}
        .jut-panel {{
            background: #f7fee7;
            border: 2px solid #a3e635;
            border-radius: 14px;
            padding: 18px 22px;
            margin-bottom: 16px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        }}
        .white-card {{
            background: #ffffff;
            border: 1.5px solid #d9f99d;
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 14px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
        }}
        .showcase-card {{
            background: #ffedd5;
            border: 1.5px solid #fed7aa;
            border-radius: 14px;
            padding: 14px;
            text-align: center;
        }}
        .showcase-circle {{
            width: 128px;
            height: 128px;
            border-radius: 50%;
            object-fit: cover;
            border: 4px solid #84cc16;
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.12);
            margin-bottom: 8px;
        }}
        .nav-box {{
            background: #f7fee7;
            border: 2px solid #a3e635;
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        }}
        .nav-title {{
            font-size: 1.1rem;
            font-weight: 800;
            color: #1e293b;
            border-bottom: 2px solid #bef264;
            padding-bottom: 8px;
            margin-bottom: 10px;
        }}
        .nav-pill {{
            background: #84cc16;
            color: #ffffff;
            font-weight: 700;
            padding: 7px 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 0.92rem;
        }}
        .nav-item {{
            color: #1e293b;
            font-size: 0.9rem;
            padding: 5px 8px;
            border-bottom: 1px dashed #d9f99d;
        }}
        .swatch {{
            display: inline-block;
            width: 22px;
            height: 22px;
            border-radius: 6px;
            border: 1.5px solid #64748b;
            vertical-align: middle;
            margin-right: 8px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

if "correction_queue" not in st.session_state:
    st.session_state.correction_queue = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None


class SimpleClothesCNN(nn.Module):
    """Компактная свёрточная сеть (CNN) для изображений 28x28."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


def closest_color_name(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    best_name = "Смешанный оттенок"
    best_dist = float("inf")
    for name, (cr, cg, cb) in NAMED_COLORS:
        dist = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def extract_colors(image: Image.Image, top_n: int = 3) -> list[dict]:
    img = ImageOps.exif_transpose(image).convert("RGB")
    w, h = img.size
    crop = img.crop((int(w * 0.18), int(h * 0.18), int(w * 0.82), int(h * 0.82)))
    small = crop.resize((64, 64))
    arr = np.asarray(small, dtype=np.float32).reshape(-1, 3)

    quantized = (np.floor(arr / 24.0) * 24.0 + 12.0).clip(0, 255).astype(int)
    counts = Counter(map(tuple, quantized))
    total = sum(counts.values())

    out = []
    for rgb, cnt in counts.most_common(top_n):
        out.append(
            {
                "name": closest_color_name(rgb),
                "hex": "#{:02X}{:02X}{:02X}".format(*rgb),
                "percent": round((cnt / total) * 100, 1),
            }
        )
    return out


def prepare_fashion_28x28(image: Image.Image) -> Image.Image:
    gray = ImageOps.exif_transpose(image).convert("L")
    gray = ImageOps.autocontrast(gray)
    return gray.resize((28, 28), Image.Resampling.BILINEAR)


def _make_synthetic_item(class_idx: int, variant: int) -> np.ndarray:
    img = Image.new("L", (28, 28), color=235)
    draw = ImageDraw.Draw(img)
    shade = 55 + ((variant * 23) % 95)

    if class_idx in (0, 2, 6):
        draw.rectangle([7, 7, 20, 23], fill=shade)
        draw.rectangle([3, 8, 7, 15], fill=shade)
        draw.rectangle([20, 8, 24, 15], fill=shade)
    elif class_idx == 1:
        draw.rectangle([8, 5, 19, 12], fill=shade)
        draw.rectangle([8, 12, 13, 24], fill=shade)
        draw.rectangle([14, 12, 19, 24], fill=shade)
    elif class_idx == 3:
        draw.rectangle([10, 5, 17, 12], fill=shade)
        draw.polygon([(10, 12), (17, 12), (22, 24), (5, 24)], fill=shade)
    elif class_idx == 4:
        draw.rectangle([6, 5, 21, 24], fill=shade)
        draw.rectangle([3, 7, 6, 18], fill=shade)
        draw.rectangle([21, 7, 24, 18], fill=shade)
    elif class_idx in (5, 7, 9):
        top_y = 9 if class_idx == 9 else 13
        draw.rectangle([5, top_y, 15, 20], fill=shade)
        draw.rectangle([5, 17, 23, 22], fill=max(20, shade - 25))
    else:
        draw.arc([9, 4, 18, 12], start=180, end=360, fill=max(30, shade - 20), width=2)
        draw.rectangle([6, 10, 21, 23], fill=shade)

    return np.asarray(img, dtype=np.uint8)


@st.cache_resource
def load_catalog_and_model():
    torch.manual_seed(42)
    model = SimpleClothesCNN(num_classes=len(FASHION_CLASSES))

    catalog_images = []
    catalog_labels = []

    try:
        from torchvision import datasets

        ds = datasets.FashionMNIST(root="./data", train=True, download=True)
        per_class = {i: 0 for i in range(len(FASHION_CLASSES))}
        for img_arr, label in zip(ds.data.numpy(), ds.targets.numpy()):
            lbl = int(label)
            if per_class[lbl] < 60:
                catalog_images.append(img_arr.astype(np.uint8))
                catalog_labels.append(lbl)
                per_class[lbl] += 1
        source_note = "Каталог сформирован на базе примеров Fashion-MNIST (28×28, grayscale)."
    except Exception:
        for lbl in range(len(FASHION_CLASSES)):
            for v in range(40):
                catalog_images.append(_make_synthetic_item(lbl, v))
                catalog_labels.append(lbl)
        source_note = (
            "Используется встроенный каталог примеров 28×28 (резервный режим без скачивания датасета)."
        )

    cat_arr = np.stack(catalog_images, axis=0)
    x_train = torch.tensor(cat_arr, dtype=torch.float32).unsqueeze(1) / 255.0
    y_train = torch.tensor(catalog_labels, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(12):
        optimizer.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        cat_probs = F.softmax(model(x_train), dim=1)

    return model, cat_arr, catalog_labels, cat_probs, source_note


def run_prediction(
    prepared_28: Image.Image,
    cat_arr: np.ndarray,
    cat_labels: list[int],
    cat_probs: torch.Tensor,
    model: nn.Module,
):
    arr_28 = np.asarray(prepared_28, dtype=np.float32)
    x_tensor = torch.tensor(arr_28 / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top3_vals, top3_idxs = torch.topk(probs, k=3)
    raw_top3 = [
        (FASHION_CLASSES[int(idx)], round(float(val) * 100, 1))
        for val, idx in zip(top3_vals, top3_idxs)
    ]

    filtered_top3 = [item for item in raw_top3 if item[1] >= 8.0]
    if not filtered_top3:
        filtered_top3 = [raw_top3[0]]

    pixel_dist = np.mean((cat_arr.astype(np.float32) - arr_28) ** 2, axis=(1, 2))
    pixel_dist_norm = (pixel_dist - pixel_dist.min()) / (pixel_dist.max() - pixel_dist.min() + 1e-8)

    prob_dist = torch.norm(cat_probs - probs.unsqueeze(0), dim=1).numpy()
    prob_dist_norm = (prob_dist - prob_dist.min()) / (prob_dist.max() - prob_dist.min() + 1e-8)

    similarity_score = 0.65 * pixel_dist_norm + 0.35 * prob_dist_norm
    nearest_indices = np.argsort(similarity_score)[:5]

    similar_items = []
    for idx in nearest_indices:
        i = int(idx)
        similar_items.append(
            {
                "image": cat_arr[i],
                "label": FASHION_CLASSES[cat_labels[i]],
                "distance": round(float(similarity_score[i]), 3),
            }
        )

    return filtered_top3, similar_items


# Шапка в стиле референса
st.markdown(
    """
    <div class="top-bar">
        <div class="brand-logo">WARDROBE<span>.AI</span></div>
        <div class="brand-tagline">Понимай стиль каталога и распознавай трендовую одежду точно</div>
    </div>
    """,
    unsafe_allow_html=True,
)

model, cat_arr, cat_labels, cat_probs, source_note = load_catalog_and_model()

main_col, nav_col = st.columns([3.3, 1], gap="medium")

with nav_col:
    st.markdown(
        f"""
        <div class="nav-box">
            <div class="nav-title">Навигация по каталогу</div>
            <div class="nav-pill">Распознавание 28×28</div>
            <div class="nav-item">Очередь исправлений: <b>{len(st.session_state.correction_queue)}</b></div>
            <div class="nav-item">Классов каталога: <b>10</b></div>
            <div class="nav-item">Похожих вещей: <b>5</b></div>
            <div class="nav-item" style="border-bottom:none; font-size:0.82rem; color:#475569; margin-top:6px;">
                {source_note}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Accuracy и macro-F1 проверяются на тестовой выборке Fashion-MNIST; "
        "обычные фото с камеры или галереи отличаются от 28×28."
    )

with main_col:
    tab_main, tab_queue = st.tabs(
        ["Каталог и распознавание", f"Очередь исправлений ({len(st.session_state.correction_queue)})"]
    )

    with tab_main:
        # Витрина трендовой одежды в круглых карточках (как в референсе)
        st.markdown('<div class="jut-panel">', unsafe_allow_html=True)
        st.markdown("#### Трендовые направления каталога")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(
                f"""
                <div class="showcase-card">
                    <img class="showcase-circle" src="{CIRCLE_1_URI}" alt="Бомбер"/>
                    <div><b>Верхняя одежда</b></div>
                    <div style="font-size:0.85rem; color:#475569;">Бомберы, куртки и пальто</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sc2:
            st.markdown(
                f"""
                <div class="showcase-card">
                    <img class="showcase-circle" src="{CIRCLE_2_URI}" alt="Платье"/>
                    <div><b>Платья и образы</b></div>
                    <div style="font-size:0.85rem; color:#475569;">Силуэты, пояса и фактуры</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sc3:
            st.markdown(
                f"""
                <div class="showcase-card">
                    <img class="showcase-circle" src="{CIRCLE_3_URI}" alt="Кроссовки"/>
                    <div><b>Обувь и стритвир</b></div>
                    <div style="font-size:0.85rem; color:#475569;">Кроссовки, ботинки и сумки</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Блок загрузки / камеры
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.subheader("1. Выбор или загрузка изображения")
        input_mode = st.radio(
            "Источник изображения:",
            ["Загрузить файл", "Камера (по желанию)"],
            horizontal=True,
        )

        uploaded_file = None
        camera_file = None

        if input_mode == "Загрузить файл":
            uploaded_file = st.file_uploader(
                "Загрузите изображение одежды (PNG, JPG, JPEG, WEBP)",
                type=None,
            )
        else:
            camera_file = st.camera_input("Сделайте снимок предмета одежды")

        raw_image = None
        file_name = "camera_capture.png"

        if uploaded_file is not None:
            file_name = uploaded_file.name
            allowed_exts = (".png", ".jpg", ".jpeg", ".webp")
            if not file_name.lower().endswith(allowed_exts):
                st.error(
                    f"Неподдерживаемый формат файла: «{file_name}». "
                    "Загрузите изображение в формате PNG, JPG, JPEG или WEBP."
                )
            else:
                try:
                    raw_image = Image.open(BytesIO(uploaded_file.getvalue()))
                except Exception:
                    st.error("Не удалось прочитать файл как изображение.")
        elif camera_file is not None:
            raw_image = Image.open(BytesIO(camera_file.getvalue()))

        if raw_image is not None:
            st.write("")
            recognize_btn = st.button("Распознать", type="primary")
            if recognize_btn or st.session_state.current_analysis is not None:
                if recognize_btn:
                    prep_28 = prepare_fashion_28x28(raw_image)
                    colors = extract_colors(raw_image, top_n=3)
                    top3, similar_items = run_prediction(
                        prep_28, cat_arr, cat_labels, cat_probs, model
                    )
                    st.session_state.current_analysis = {
                        "file_name": file_name,
                        "raw_image": ImageOps.exif_transpose(raw_image).convert("RGB"),
                        "prep_28": prep_28,
                        "colors": colors,
                        "top3": top3,
                        "similar_items": similar_items,
                    }
        st.markdown("</div>", unsafe_allow_html=True)

        if raw_image is not None and st.session_state.current_analysis is not None:
            res = st.session_state.current_analysis

            st.markdown('<div class="white-card">', unsafe_allow_html=True)
            st.subheader("2. Исходное фото, версия 28×28 и цвета")
            c1, c2, c3 = st.columns([1.2, 1, 1.3], gap="medium")

            with c1:
                st.markdown("**Исходное изображение**")
                st.image(res["raw_image"], use_container_width=True)

            with c2:
                st.markdown("**Подготовленное (28×28, grayscale)**")
                st.image(
                    res["prep_28"].resize((196, 196), Image.Resampling.NEAREST),
                    caption="Масштаб 196×196 для просмотра (вход: 28×28)",
                    use_container_width=True,
                )

            with c3:
                st.markdown("**Основные цвета предмета**")
                for col_info in res["colors"]:
                    st.markdown(
                        f"<div style='margin-bottom:8px;'>"
                        f"<span class='swatch' style='background:{col_info['hex']};'></span>"
                        f"<b>{col_info['name']}</b> — <code>{col_info['hex']}</code> ({col_info['percent']}%)"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="white-card">', unsafe_allow_html=True)
            st.subheader("3. Предполагаемые категории")
            for rank, (cat_name, pct) in enumerate(res["top3"], start=1):
                st.write(f"**{rank}. {cat_name}** — `{pct}%`")
                st.progress(min(max(pct / 100.0, 0.0), 1.0))

            st.markdown("---")
            st.markdown("**Ответ неверный? Отправьте пример в очередь исправлений:**")
            col_err1, col_err2 = st.columns([2, 1])
            with col_err1:
                true_label = st.selectbox(
                    "Выберите правильную категорию:",
                    FASHION_CLASSES,
                    key="true_label_select",
                )
            with col_err2:
                st.write("")
                st.write("")
                if st.button("Отметить неверный ответ"):
                    st.session_state.correction_queue.append(
                        {
                            "Файл": res["file_name"],
                            "Предсказано (Топ-1)": f"{res['top3'][0][0]} ({res['top3'][0][1]}%)",
                            "Правильная категория": true_label,
                            "Цвет": res["colors"][0]["name"] if res["colors"] else "—",
                        }
                    )
                    st.success("Пример добавлен в «Очередь исправлений»!")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="white-card">', unsafe_allow_html=True)
            st.subheader("4. Пять похожих изображений из обучающего каталога")
            sim_cols = st.columns(5)
            for idx, item in enumerate(res["similar_items"]):
                with sim_cols[idx]:
                    st.image(
                        Image.fromarray(item["image"]).resize((140, 140), Image.Resampling.NEAREST),
                        use_container_width=True,
                    )
                    st.caption(f"**{item['label']}**\nСхожесть: `{item['distance']}`")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_queue:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.subheader("Очередь исправлений для дообучения и проверки")
        if not st.session_state.correction_queue:
            st.info(
                "Очередь пока пуста. Нажмите «Отметить неверный ответ» после распознавания, чтобы сохранить пример."
            )
        else:
            st.dataframe(st.session_state.correction_queue, use_container_width=True)
            if st.button("Очистить очередь"):
                st.session_state.correction_queue = []
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
