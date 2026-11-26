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
    page_icon="👗",
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


def build_clothing_bg_data_uri() -> str:
    """Создаёт мягкий фоновый узор с иллюстрациями одежды в SVG."""
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="320" height="320" viewBox="0 0 320 320">
      <rect width="320" height="320" fill="#fdf4ff"/>
      <!-- Худи / футболка слева сверху -->
      <g transform="translate(26,24)" opacity="0.28">
        <path d="M24 10 L42 6 L52 16 L62 6 L80 10 L96 32 L78 44 L72 34 L72 86 L28 86 L28 34 L22 44 L4 32 Z"
              fill="#ec4899" stroke="#be185d" stroke-width="2.5" stroke-linejoin="round"/>
      </g>
      <!-- Платье справа сверху -->
      <g transform="translate(185,32)" opacity="0.26">
        <path d="M36 8 L54 8 L58 24 L82 78 L10 78 L32 24 Z"
              fill="#8b5cf6" stroke="#5b21b6" stroke-width="2.5" stroke-linejoin="round"/>
      </g>
      <!-- Куртка / пальто по центру -->
      <g transform="translate(112,126)" opacity="0.25">
        <path d="M24 8 L56 8 L76 28 L66 42 L58 34 L58 92 L22 92 L22 34 L14 42 L4 28 Z"
              fill="#3b82f6" stroke="#1d4ed8" stroke-width="2.5" stroke-linejoin="round"/>
        <line x1="40" y1="8" x2="40" y2="92" stroke="#1d4ed8" stroke-width="2"/>
      </g>
      <!-- Кроссовок снизу слева -->
      <g transform="translate(28,214)" opacity="0.26">
        <path d="M8 46 L26 46 L42 22 L56 28 L68 42 L88 46 L88 56 L8 56 Z"
              fill="#f59e0b" stroke="#b45309" stroke-width="2.5" stroke-linejoin="round"/>
      </g>
      <!-- Сумка снизу справа -->
      <g transform="translate(204,206)" opacity="0.26">
        <path d="M24 28 C24 10 56 10 56 28" fill="none" stroke="#db2777" stroke-width="3.5"/>
        <rect x="12" y="28" width="56" height="46" rx="8"
              fill="#f472b6" stroke="#be185d" stroke-width="2.5"/>
      </g>
    </svg>
    """.strip()
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


BG_URI = build_clothing_bg_data_uri()

st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.78), rgba(250, 245, 255, 0.84)), url("{BG_URI}");
            background-repeat: repeat;
            background-size: 320px 320px;
        }}
        .hero-banner {{
            background: linear-gradient(135deg, #4f46e5 0%, #db2777 55%, #ea580c 100%);
            padding: 22px 26px;
            border-radius: 18px;
            color: #ffffff;
            margin-bottom: 18px;
            box-shadow: 0 10px 24px rgba(79, 70, 229, 0.22);
        }}
        .hero-banner h1 {{
            color: #ffffff;
            margin: 0 0 6px 0;
            font-size: 1.9rem;
        }}
        .hero-banner p {{
            margin: 0;
            opacity: 0.96;
            font-size: 0.98rem;
        }}
        .card-box {{
            background: rgba(255, 255, 255, 0.94);
            backdrop-filter: blur(6px);
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        }}
        .swatch {{
            display: inline-block;
            width: 24px;
            height: 24px;
            border-radius: 7px;
            border: 1px solid #cbd5e1;
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
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
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
    """Создаёт наглядный шаблон 28x28 для выбранного класса каталога."""
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

    # Быстрое обучение на выборке каталога, чтобы сеть выделяла признаки классов
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

    # Убираем совсем слабые случайные варианты ниже 8%, оставляя минимум главный вариант
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
        <h1>👗 Помощник каталога одежды — Fashion-MNIST</h1>
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
        2. Предполагаемые категории (**Топ-3**, слабые шумовые варианты ниже 8% отсекаются).
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
    st.subheader("Очередь исправлений для дообучения и проверки")
    if not st.session_state.correction_queue:
        st.info("Очередь пока пуста. Нажмите «Отметить неверный ответ» после распознавания, чтобы сохранить пример.")
    else:
        st.dataframe(st.session_state.correction_queue, use_container_width=True)
        if st.button("Очистить очередь"):
            st.session_state.correction_queue = []
            st.rerun()
