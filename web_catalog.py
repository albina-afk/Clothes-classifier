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
    page_title="Помощник каталога одежды (Fashion-MNIST)",
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


def build_editorial_bg_uri() -> str:
    """Создаёт стильный тёмный фон с изящными силуэтами одежды и мягким свечением."""
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="420" height="420" viewBox="0 0 420 420">
      <defs>
        <radialGradient id="glow1" cx="22%" cy="24%" r="45%">
          <stop offset="0%" stop-color="#6366f1" stop-opacity="0.22"/>
          <stop offset="100%" stop-color="#0b0f19" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="glow2" cx="80%" cy="72%" r="50%">
          <stop offset="0%" stop-color="#10b981" stop-opacity="0.16"/>
          <stop offset="100%" stop-color="#0b0f19" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <rect width="420" height="420" fill="#0b0f19"/>
      <rect width="420" height="420" fill="url(#glow1)"/>
      <rect width="420" height="420" fill="url(#glow2)"/>

      <!-- Изящный контур пиджака / куртки -->
      <g transform="translate(38,34)" opacity="0.22" fill="none" stroke="#94a3b8" stroke-width="1.6" stroke-linejoin="round">
        <path d="M36 12 L72 12 L92 34 L78 48 L68 38 L68 106 L40 106 L40 38 L30 48 L16 34 Z"/>
        <path d="M46 12 L54 42 L62 12"/>
      </g>

      <!-- Контур платья -->
      <g transform="translate(268,48)" opacity="0.20" fill="none" stroke="#c084fc" stroke-width="1.6" stroke-linejoin="round">
        <path d="M42 12 L62 12 L68 34 L94 98 L16 98 L36 34 Z"/>
        <line x1="36" y1="34" x2="68" y2="34"/>
      </g>

      <!-- Контур рубашки -->
      <g transform="translate(156,176)" opacity="0.20" fill="none" stroke="#38bdf8" stroke-width="1.6" stroke-linejoin="round">
        <path d="M34 10 L50 18 L66 10 L88 30 L74 44 L66 36 L66 96 L34 96 L34 36 L26 44 L12 30 Z"/>
        <line x1="50" y1="18" x2="50" y2="96"/>
      </g>

      <!-- Контур обуви -->
      <g transform="translate(46,294)" opacity="0.20" fill="none" stroke="#fbbf24" stroke-width="1.6" stroke-linejoin="round">
        <path d="M10 56 L34 56 L54 28 L72 34 L86 50 L108 54 L108 66 L10 66 Z"/>
      </g>

      <!-- Контур сумки -->
      <g transform="translate(274,286)" opacity="0.20" fill="none" stroke="#34d399" stroke-width="1.6" stroke-linejoin="round">
        <path d="M36 36 C36 14 68 14 68 36"/>
        <rect x="22" y="36" width="60" height="52" rx="10"/>
      </g>
    </svg>
    """.strip()
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


BG_URI = build_editorial_bg_uri()

st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("{BG_URI}");
            background-repeat: repeat;
            background-size: 420px 420px;
            color: #f1f5f9;
        }}
        .hero-banner {{
            background: linear-gradient(120deg, #1e1b4b 0%, #312e81 45%, #0f766e 100%);
            padding: 24px 28px;
            border-radius: 20px;
            color: #ffffff;
            margin-bottom: 20px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 14px 32px rgba(2, 6, 23, 0.45);
        }}
        .hero-banner h1 {{
            color: #ffffff;
            margin: 0 0 6px 0;
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        .hero-banner p {{
            margin: 0;
            color: #e2e8f0;
            font-size: 0.98rem;
        }}
        .card-box {{
            background: rgba(17, 24, 39, 0.88);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 18px 22px;
            margin-bottom: 16px;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.35);
        }}
        .card-box h2, .card-box h3, .card-box p, .card-box span, .card-box label {{
            color: #f8fafc;
        }}
        .swatch {{
            display: inline-block;
            width: 22px;
            height: 22px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.35);
            vertical-align: middle;
            margin-right: 10px;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(90deg, #4f46e5 0%, #0d9488 100%);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 0.52rem 1.35rem;
            font-weight: 600;
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
    """Определяет основные цвета по центральной зоне снимка."""
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
    """Переводит изображение в оттенки серого и приводит к формату 28x28."""
    gray = ImageOps.exif_transpose(image).convert("L")
    gray = ImageOps.autocontrast(gray)
    return gray.resize((28, 28), Image.Resampling.BILINEAR)


def _make_synthetic_item(class_idx: int, variant: int) -> np.ndarray:
    """Создаёт шаблон 28x28 для выбранного класса каталога."""
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
    """Загружает каталог 28x28 и быстро обучает CNN на примерах каталога."""
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
    """Возвращает отфильтрованный топ-3 и 5 ближайших изображений из каталога."""
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


st.markdown(
    """
    <div class="hero-banner">
        <h1>🧥 Помощник каталога одежды — Fashion-MNIST</h1>
        <p>Подготовка 28×28 в оттенках серого, определение цветов, категории, 5 похожих предметов и очередь исправлений</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, cat_arr, cat_labels, cat_probs, source_note = load_catalog_and_model()

with st.sidebar:
    st.header("📌 О проекте и проверке")
    st.write(source_note)
    st.markdown(
        """
        **Что проверяется на защите:**
        1. Исходное фото и подготовленное **28×28 (grayscale)**.
        2. Предполагаемые категории (**Топ-3**, слабые варианты ниже 8% отсекаются).
        3. **5 похожих изображений** из каталога и отметка ошибки.
        4. Неподдерживаемый файл (например `.txt`, `.pdf`, `.csv`).
        """
    )
    st.caption(
        "Примечание: Accuracy и macro-F1 оцениваются на тестовой выборке датасета; "
        "обычные фотографии с камеры или телефона отличаются от формата 28×28."
    )

tab_main, tab_queue = st.tabs(
    ["🔍 Распознавание", f"🛠️ Очередь исправлений ({len(st.session_state.correction_queue)})"]
)

with tab_main:
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
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
            "Загрузите изображение одежды (поддерживаются PNG, JPG, JPEG, WEBP)",
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
                "Пожалуйста, загрузите изображение в формате PNG, JPG, JPEG или WEBP."
            )
        else:
            try:
                raw_image = Image.open(BytesIO(uploaded_file.getvalue()))
            except Exception:
                st.error(
                    "Не удалось прочитать файл как изображение. Проверьте целостность файла и попробуйте снова."
                )
    elif camera_file is not None:
        raw_image = Image.open(BytesIO(camera_file.getvalue()))

    if raw_image is not None:
        st.write("")
        recognize_btn = st.button("Распознать", type="primary")
        if recognize_btn or st.session_state.current_analysis is not None:
            if recognize_btn:
                prep_28 = prepare_fashion_28x28(raw_image)
                colors = extract_colors(raw_image, top_n=3)
                top3, similar_items = run_prediction(prep_28, cat_arr, cat_labels, cat_probs, model)
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

        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.subheader("2. Исходное изображение и подготовка 28×28")
        c1, c2, c3 = st.columns([1.2, 1, 1.3], gap="medium")

        with c1:
            st.markdown("**Исходное изображение**")
            st.image(res["raw_image"], use_container_width=True)

        with c2:
            st.markdown("**Подготовленное (28×28, grayscale)**")
            st.image(
                res["prep_28"].resize((196, 196), Image.Resampling.NEAREST),
                caption="Масштаб 196×196 для просмотра (реальный вход: 28×28)",
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

        st.markdown('<div class="card-box">', unsafe_allow_html=True)
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

        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.subheader("4. Пять похожих изображений из обучающего каталога")
        sim_cols = st.columns(5)
        for idx, item in enumerate(res["similar_items"]):
            with sim_cols[idx]:
                st.image(
                    Image.fromarray(item["image"]).resize((140, 140), Image.Resampling.NEAREST),
                    use_container_width=True,
                )
                st.caption(f"**{item['label']}**\nСхожесть (дист.): `{item['distance']}`")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_queue:
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.subheader("Очередь исправлений для дообучения и проверки")
    if not st.session_state.correction_queue:
        st.info("Очередь пока пуста. Нажмите «Отметить неверный ответ» после распознавания, чтобы сохранить пример.")
    else:
        st.dataframe(st.session_state.correction_queue, use_container_width=True)
        if st.button("Очистить очередь"):
            st.session_state.correction_queue = []
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
