from transformers import AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from PIL import Image, ImageDraw, ImageFont, ImageColor
from difflib import SequenceMatcher
from pytesseract import Output
from faker import Faker
import numpy as np
import difflib
import pytesseract
import spacy
import cv2
import ast
import re

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

font_path = "C:/Users/User/Documents/VKR/ner/final_scripts/font/Arial.ttf"

colors = {
    "ORG": "#A9A9A9",        # серый — организация
    "PER": "#7FFFD4",        # аквамарин — человек
    "LOC": "#FFFF00",        # жёлтый — геолокация
    "DATE": "#87CEFA",       # голубой — дата
    "SNILS_NUM": "#DDA0DD",  # сиреневый — СНИЛС
    "ACC_NUM": "#808000",    # оливковый — счёт
    "PASP_NUM": "#FA8072",   # красноватый — паспорт
    "PHONE_NUM": "#00CED1",  # бирюзовый — телефон
    "EMAIL": "#F67DE3",      # розовый — почта
    "KPP_NUM": "#ADFF2F",    # салатовый — КПП
    "INN_NUM": "#FFA500",    # оранжевый — ИНН
    "IP": "#7DF6D9"          # светло-бирюзовый — IP
}

fake = Faker(locale=['ru_RU'])

base_model_path = "ru_core_news_sm"
base_spacy_model = spacy.load(base_model_path)

my_model_path = "model/my_ner_model_10"
my_spacy_model = spacy.load(my_model_path)


model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

model_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor_vl = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

# model_name = "Qwen/Qwen2.5-7B-Instruct"
# model_instruct = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
tokenizer_instruct = AutoTokenizer.from_pretrained(model_name)


def get_text_angle(image):
    # Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Убираем шумы и находим границы текста
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Находим линии методом Хафа
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)

    angles = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

    # Берем медианный угол (чтобы избежать выбросов)
    if angles:
        median_angle = np.median(angles)
    else:
        median_angle = 0

    return median_angle


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Создаем матрицу поворота
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated_image


def auto_correct_skew(image):
    
    if image is None:
        print("Ошибка: изображение не загружено!")
        return

    # Определяем угол наклона текста
    angle = get_text_angle(image)
    #print(f"Определенный угол наклона: {angle:.2f}°")

    # Поворачиваем изображение
    corrected_image = rotate_image(image, -angle)

    return corrected_image


def replace_entity(entity_type):

    if entity_type in ["PER", "PERSON"]:
        return fake.name().split()[1]+ ' ' + fake.name().split()[0]
        
    elif entity_type in ["EMAIL", "EMAIL_ADDRESS"]:
        return fake.email()
        
    elif entity_type in ["LOC", "LOCATION"]:
        return fake.city()
        
    elif entity_type in ["IP", "IP_ADDRESS"]:
        return fake.ipv4()
        
    elif entity_type in ["ORG"]:
        return fake.company()
        
    elif entity_type in ["PHONE_NUM", "PHONE_NUMBER"]:
        return fake.phone_number()

    elif entity_type in ["KPP_NUM"]:
        return fake.kpp()

    elif entity_type in ["INN_NUM"]:
        return fake.individuals_inn()

    elif entity_type in ["PASP_NUM"]:
        return fake.passport_number()

    elif entity_type in ["SNILS_NUM"]:
        return fake.snils()

    elif entity_type in ["ACC_NUM"]:
        return fake.checking_account()
        
    elif entity_type in ["DATE"]:
        return fake.date()
        
    else:
        return '[NO_LABEL_REPLACE]'
    

def get_entity_bboxes(words_bbox_by_line, base_spacy_model, my_spacy_model):
    results = []

    for line in words_bbox_by_line:
        
        line_text = ' '.join([w['text'] for w in line])
    
        line_nlp_base = base_spacy_model(line_text)
        base_ent = [(i, i.label_, i.start_char, i.end_char) for i in line_nlp_base.ents if i.label_ in ["ORG", "PER", "LOC"]]
        
        line_nlp_my = my_spacy_model(line_text)
        my_ent = [(i, i.label_, i.start_char, i.end_char) for i in line_nlp_my.ents if i.label_ not in ["ORG", "PER", "LOC"]]
    
        ent_list = base_ent + my_ent

        # Отдельные слова в строке (без пробелов)
        words_in_line = [w['text'] for w in line]
        word_spans = []
        offset = 0

        # Определим начальную и конечную позиции каждого слова в строке
        for word in words_in_line:
            start = offset
            end = offset + len(word)
            word_spans.append((start, end))
            offset = end + 1  # +1 за пробел

        for ent in ent_list:
            ent_start = ent[2]
            ent_end = ent[3]
            ent_words = []

            # Находим слова, которые входят в диапазон сущности
            for i, (start, end) in enumerate(word_spans):
                if end > ent_start and start < ent_end:
                    ent_words.append(line[i])

            if not ent_words:
                continue

            # Получаем объединённый bbox
            x1 = min(w['bbox'][0] for w in ent_words)
            y1 = min(w['bbox'][1] for w in ent_words)
            x2 = max(w['bbox'][2] for w in ent_words)
            y2 = max(w['bbox'][3] for w in ent_words)

            results.append({
                'type': ent[1],
                'text': ent[0],
                'bbox': [x1, y1, x2, y2]
            })

    return results


def get_fitting_font_size(text, bbox, font_path, max_font_size=60, min_font_size=8):
    """
    Возвращает максимальный размер шрифта, при котором текст влезает в bbox.
    """
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1

    # Пустое изображение для измерения текста
    temp_img = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(temp_img)

    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        if text_width <= box_width and text_height <= box_height:
            return font_size

    return min_font_size  # если не влезает — минимальный


def get_contrast_text_color(rgb: tuple[int, int, int]) -> str:
    """
    Определяет, какой цвет текста ('black' или 'white') будет лучше читаться на фоне указанного RGB-цвета.

    :param rgb: Кортеж из трёх чисел (R, G, B), каждое от 0 до 255
    :return: 'black' если фон светлый, 'white' если фон тёмный
    """
    r, g, b = rgb[:3]  # используем только R, G, B
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if brightness > 186 else 'white'


def mask_image(image, image_path, entities):
    
    draw = ImageDraw.Draw(image)

    # Проходим по каждой сущности и заменяем её текст
    for entity in entities:
        
        entity_text = replace_entity(entity['type'])  # Заменяем текст на основе типа сущности

        x1, y1, x2, y2 = entity['bbox']
        width = x2 - x1

        # Получаем цвет пикселя в центре прямоугольника
        center_x = (x1 + 1)
        center_y = (y1 + 1) 
        color = image.getpixel((center_x, center_y))  # Получаем цвет в центре

        # Заливка прямоугольника (можно сделать белым или другим цветом)
        print(color)
        draw.rectangle([x1, y1, x2, y2], fill=color) #color

        font_size = get_fitting_font_size(entity_text, [x1, y1, x2, y2], font_path)
        font = ImageFont.truetype(font_path, font_size)

        text_color = get_contrast_text_color(color)

        # Пишем заменённый текст
        draw.text((x1, y1), entity_text, font=font, fill=text_color)

    output_path = image_path.split('.')[0] + '_final' + '.' + image_path.split('.')[1]
    image.save(output_path)
    print(f"Сохранено в {output_path}")

    return image


def get_words_with_boxes_by_lines(image):
    # image_to_data неверно разбивает по строкам, а image_to_string верно, поэтому используем комбинацию
    data = pytesseract.image_to_data(image, lang="rus+eng", output_type=Output.DICT)
    text = pytesseract.image_to_string(image, lang="rus+eng")
    
    lines = text.strip().split('\n')
    words_by_line = [line.strip().split() for line in lines if line.strip()]

    text_by_lines = [i for i in lines if i!='']

    word_data = []

    # Очистим слова из data
    raw_words = data['text']
    boxes = zip(data['left'], data['top'], data['width'], data['height'])
    word_boxes = [(word.strip(), (x, y, x + w, y + h))
                  for word, (x, y, w, h) in zip(raw_words, boxes)
                  if word.strip() != ""]

    i = 0
    for line_words in words_by_line:
        current_line = []
        for word in line_words:
            # Сопоставляем текущее слово с data (в порядке появления)
            while i < len(word_boxes):
                data_word, bbox = word_boxes[i]
                i += 1
                # Проверка на совпадение по содержанию (можно использовать .lower() или fuzzy matching при необходимости)
                if data_word == word or data_word.strip('.,:;!?()[]') == word.strip('.,:;!?()[]'):
                    current_line.append({'text': word, 'bbox': bbox})
                    break
        if current_line:
            word_data.append(current_line)

    return text_by_lines, word_data


def ocr_draw(image, image_path, words_bbox_by_line):

    draw = ImageDraw.Draw(image)
    
    # Параметры отступа
    padding = 2
    
    # Обводим каждый bbox красной линией
    for line in words_bbox_by_line:
        for word in line:
            x1, y1, x2, y2 = word['bbox']
            padded_bbox = (x1 - padding, y1 - padding, x2 + padding, y2 + padding)
            draw.rectangle(padded_bbox, outline='red', width=1)
    
    output_path = image_path.split('.')[0] + '_ocr' + '.' + image_path.split('.')[1]
    image.save(output_path)
    print(f"Сохранено в {output_path}")

    return image


def ner_draw(image, image_path, entities):
    default_color = "#AAAAAA"

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    padding = 4
    opacity = 255
    
    for ent in entities:
        label = ent["type"]
        x1, y1, x2, y2 = ent["bbox"]
    
        # Отступы
        x1_pad = x1 - padding
        y1_pad = y1 - padding
        x2_pad = x2 + padding
        y2_pad = y2 + padding
    
        # Получение цвета
        color_hex = colors.get(label, default_color)
        color_rgb = ImageColor.getrgb(color_hex)
        
        # Нарисовать только контур прямоугольника (без заливки)
        draw.rectangle([x1_pad, y1_pad, x2_pad, y2_pad], outline=color_hex, width=2)
    
        # Подпись
        bbox_label = font.getbbox(label)
        text_width = bbox_label[2] - bbox_label[0]
        text_height = bbox_label[3] - bbox_label[1]
        text_x = (x1_pad + x2_pad - text_width) / 2
        text_y = y1_pad - text_height - 2
    
        # Фон под подпись
        draw.rectangle(
            [text_x - 2, text_y - 1, text_x + text_width + 2, text_y + text_height + 1],
            fill=color_hex
        )
        # Нарисовать текст
        draw.text((text_x, text_y), label, fill="black", font=font)
    
    # Сохранение
    output_path = image_path.split('.')[0] + '_ner' + '.' + image_path.split('.')[1]
    image.save(output_path)
    print(f"Сохранено в {output_path}")

    return image


def preprocessing(image):
    # Конвертируем в numpy
    image_np = np.array(image)

    # Преобразуем в BGR, если нужно (OpenCV работает в BGR)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # # Применяем коррекцию наклона, если нужно
    corrected_image = auto_correct_skew(image_cv)

    # Преобразуем изображение в оттенки серого для обработки
    gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    return rgb


def normalize_text(text):
    return text.lower().replace(" ", "")

def get_combined_bbox(bboxes):
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]


def group_entities(words_bbox_by_line, entities):
    # Преобразуем список сущностей в словарь по нормализованному тексту
    ent_map = {normalize_text(text): label for text, label in entities}

    result = []

    for line in words_bbox_by_line:
        i = 0
        while i < len(line):
            word_i = line[i]['text']
            acc_text = word_i
            acc_bbox = [line[i]['bbox']]
            j = i + 1

            found_type = ent_map.get(normalize_text(word_i))
            current_type = found_type

            while j < len(line):
                word_j = line[j]['text']
                next_text = acc_text + ' ' + word_j
                next_type = ent_map.get(normalize_text(next_text))

                # Если объединённый текст есть в entities — обновим
                if next_type:
                    acc_text = next_text
                    acc_bbox.append(line[j]['bbox'])
                    current_type = next_type
                    j += 1
                else:
                    # Проверим: может быть это два отдельных слова из одной сущности?
                    if found_type and ent_map.get(normalize_text(word_j)) == found_type:
                        acc_text += ' ' + word_j
                        acc_bbox.append(line[j]['bbox'])
                        j += 1
                    else:
                        break

            if current_type:
                result.append({
                    'type': current_type,
                    'text': acc_text,
                    'bbox': get_combined_bbox(acc_bbox)
                })
                i = j
            else:
                i += 1

    return result


def ner_predict_llm(model, tokenizer, text):
    
    prompt = f"""
    Найди и перечисли все сущности следующих типов:
    ORG — организация (компания, учреждение)
    PER — персона (ФИО, имя и фамилия, фамилия и т.п.)
    LOC — географическое место (город, страна и т.п.)
    DATE — дата (в любом формате)
    SNILS_NUM — номер СНИЛС (формат: XXX-XXX-XXX XX)
    ACC_NUM — номер банковского счёта (обычно 20 цифр)
    PASP_NUM — номер паспорта (обычно 10 цифр)
    PHONE_NUM — номер телефона (в любом читаемом формате)
    EMAIL — адрес электронной почты
    KPP_NUM — КПП (обычно 9 цифр)
    INN_NUM — ИНН (для физлиц 12 цифр, для юрлиц — 10 цифр)
    IP — IP-адрес (в формате IPv4, например, 192.168.0.1)
    
    Верни ответ в виде списка сущностей с указанием их типа и значением. Не добавляй лишнего текста и символов. Формат вывода:
    [('Алексеева Фёкла Романовна', 'PER'),
     ('Новосибирской области', 'LOC'),
     ('ЗАО «Никитина Журавлев»', 'ORG'),
     ('2 мая 1996 года', 'DATE'),
     ('02.06.1997', 'DATE'),
     ...]
    
    Текст для анализа:
    {text}
    """
    
    messages = [
        {"role": "system", "content": "Ты — система извлечения именованных сущностей из текста."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def correct_ner_llm(model, tokenizer, reserence_text, ner_text):
    
    prompt = f"""
    Ты — эксперт по обработке текста и извлечению сущностей (NER).
    Ниже представлен список извлечённых сущностей с типами, текстами и координатами (bbox), полученный от модели.
    Также дан эталонный текст (reference), из которого они были извлечены.
    Твоя задача — оценить, насколько корректно извлечены сущности:
    - Совпадает ли тип сущности?
    - Насколько точно извлечён текст сущности?
    Для каждой сущности оцени корректность от 0.0 до 1.0.
    Формат ответа:
    [
      {{"text": "Сбербанк", "type": "ORG", "score": 1.0}},
      {{"text": "На Столе Кошка", "type": "PER", "score": 0.0}},
      ...
    ]
    Оригинальный (reference) текст:
    {reserence_text}
    Распознанные сущности:
    {ner_text}
    """
    
    messages = [
        {"role": "system", "content": "Ты — система извлечения именованных сущностей из текста."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def correct_ner_vllm(model, processor, image_path, reserence_text, ner_text):
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}",
                },
            {"type": "text", "text": f"""
            Ты — эксперт по обработке текста и извлечению сущностей (NER).
            Ниже представлен список извлечённых сущностей с типами, текстами и координатами (bbox), полученный от модели.
            Также дан эталонный текст (reference), из которого они были извлечены.
            Твоя задача — оценить, насколько корректно извлечены сущности:
            - Совпадает ли тип сущности?
            - Насколько точно извлечён текст сущности?
            Для каждой сущности оцени корректность от 0.0 до 1.0.
            Формат ответа:
            [
              {{"text": "Сбербанк", "type": "ORG", "score": 1.0}},
              {{"text": "На Столе Кошка", "type": "PER", "score": 0.0}},
              ...
            ]
            Оригинальный (reference) текст:
            {reserence_text}
            Распознанные сущности:
            {ner_text}
            """},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


def correct_image_vllm(model, processor, image_path, ocr_text):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}",
                },
            {"type": "text", "text": f"""
            Ты — интеллектуальная модель, способная анализировать изображения и исправлять ошибки в результатах OCR. Твоя задача — сравнить распознанный текст с изображением и предложить точную версию текста без ошибок.
            Вот текст, полученный с помощью OCR:
            {str(ocr_text)}
            Этот текст может содержать ошибки, например, неверные символы, пропущенные буквы или лишние пробелы.
            Используй изображение, чтобы восстановить и дополнить при необходимости исходный текст.
            Верни результат без лишних пояснений.
            """},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


def align_text_with_bboxes(words_bbox_by_line, corrected_ocr_list):
    aligned = []
    
    for ocr_words, corrected_line in zip(words_bbox_by_line, corrected_ocr_list):
        corrected_tokens = corrected_line.split()
        original_texts = [w['text'] for w in ocr_words]

        matcher = SequenceMatcher(None, original_texts, corrected_tokens)
        result_line = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    result_line.append({'text': corrected_tokens[j], 'bbox': ocr_words[i]['bbox']})
            elif tag in ('replace', 'delete', 'insert'):
                # пытаемся выровнять вручную: обрезать/объединить bbox-ы
                original_slice = ocr_words[i1:i2]
                corrected_slice = corrected_tokens[j1:j2]

                # упрощённо: если длины равны, сопоставим напрямую
                if len(original_slice) == len(corrected_slice):
                    for o, c in zip(original_slice, corrected_slice):
                        result_line.append({'text': c, 'bbox': o['bbox']})
                elif len(corrected_slice) == 1:
                    # объединяем bbox-ы
                    if original_slice:
                        x0 = min(w['bbox'][0] for w in original_slice)
                        y0 = min(w['bbox'][1] for w in original_slice)
                        x1 = max(w['bbox'][2] for w in original_slice)
                        y1 = max(w['bbox'][3] for w in original_slice)
                        result_line.append({'text': corrected_slice[0], 'bbox': (x0, y0, x1, y1)})
                else:
                    # если ничего не сходится — добавим bbox-ы наугад, чтобы сохранить длину
                    for c in corrected_slice:
                        result_line.append({'text': c, 'bbox': (0, 0, 0, 0)})
        aligned.append(result_line)
    
    return aligned


def ner_predict_llm(model, tokenizer, text):
    
    prompt = f"""
    Найди и перечисли все сущности следующих типов:
    ORG — организация (компания, учреждение)
    PER — персона (ФИО, имя и фамилия, фамилия и т.п.)
    LOC — географическое место (город, страна и т.п.)
    DATE — дата (в любом формате)
    SNILS_NUM — номер СНИЛС (формат: XXX-XXX-XXX XX)
    ACC_NUM — номер банковского счёта (обычно 20 цифр)
    PASP_NUM — номер паспорта (обычно 10 цифр)
    PHONE_NUM — номер телефона (в любом читаемом формате, включая +7 и 8)
    EMAIL — адрес электронной почты
    KPP_NUM — КПП (обычно 9 цифр)
    INN_NUM — ИНН (для физлиц 12 цифр, для юрлиц — 10 цифр)
    IP — IP-адрес (в формате IPv4, например, 192.168.0.1)
    
    Верни ответ в виде списка сущностей с указанием их типа и значением. Сущности должны остаться в первоначальном виде (важно!). Формат вывода:
    [('Алексеева Фёкла Романовна', 'PER'),
     ('Новосибирской области', 'LOC'),
     ('ЗАО «Никитина Журавлев»', 'ORG'),
     ...]
    
    Текст для анализа:
    {text}
    """
    
    messages = [
        {"role": "system", "content": "Ты — система извлечения именованных сущностей из текста."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def normalize_text(text: str) -> str:
    return re.sub(r'[\s\.,;:!?«»"\'()-]+', '', text.lower())


def filter_entities_by_score(entities, entities_corrected, threshold=0.5):
    filtered = []
    for ent_corr in entities_corrected:
        if ent_corr["score"] >= threshold:
            norm_corr_text = normalize_text(ent_corr["text"])
            for ent in entities:
                if ent["type"] == ent_corr["type"] and normalize_text(ent["text"].text) == norm_corr_text:
                    filtered.append({
                        "text": ent["text"],
                        "type": ent["type"],
                        "bbox": ent["bbox"]
                    })
                    break
    return filtered

def get_merged_bbox(tokens: list[dict]) -> list[int]:
    x0 = min(t['bbox'][0] for t in tokens)
    y0 = min(t['bbox'][1] for t in tokens)
    x1 = max(t['bbox'][2] for t in tokens)
    y1 = max(t['bbox'][3] for t in tokens)
    return [x0, y0, x1, y1]

def build_normalized_entities(entity_list: list[tuple[str, str]]) -> list[dict]:
    return [{
        'type': ent_type,
        'text': ent_text,
        'norm_text': normalize_text(ent_text)
    } for ent_text, ent_type in entity_list]

def match_entities_in_ocr(
    words_bbox_by_line: list[list[dict]],
    entities: list[tuple[str, str]],
    fuzzy_threshold: float = 0.9
) -> list[dict]:
    normalized_entities = build_normalized_entities(entities)
    matched_entities = []
    used_indices = set()

    for line_idx, line in enumerate(words_bbox_by_line):
        for i in range(len(line)):
            token_seq = []
            for j in range(i, len(line)):
                token_seq.append(line[j])
                joined_text = ' '.join(tok['text'] for tok in token_seq)
                joined_norm = normalize_text(joined_text)

                for ent_idx, ent in enumerate(normalized_entities):
                    if ent_idx in used_indices:
                        continue  # уже использовали эту сущность

                    sim = SequenceMatcher(None, joined_norm, ent['norm_text']).ratio()
                    if sim >= fuzzy_threshold:
                        matched_entities.append({
                            'type': ent['type'],
                            'text': joined_text,
                            'bbox': get_merged_bbox(token_seq)
                        })
                        used_indices.add(ent_idx)
                        break  # не продолжаем расширять последовательность
    return matched_entities


def main(image, image_path, model_ner='qwen', correct_ocr=True, correct_ner=True): 

    rgb = preprocessing(image)

    text_by_lines, words_bbox_by_line = get_words_with_boxes_by_lines(rgb)
    
    # Корректировка OCR VLLM
    if correct_ocr:
        corrected_ocr_text_str = correct_image_vllm(model_vl, processor_vl, image_path, text_by_lines)
        corrected_ocr_text_list = ast.literal_eval(corrected_ocr_text_str)
        words_bbox_by_line = align_text_with_bboxes(words_bbox_by_line, corrected_ocr_text_list)

    print(text_by_lines)
    
    ocr_image = ocr_draw(image.copy(), image_path, words_bbox_by_line)

    if model_ner == 'qwen':
        entities_base = ner_predict_llm(model_vl, tokenizer_instruct, ' '.join(text_by_lines))
        entities_list = ast.literal_eval(entities_base)
        entities = match_entities_in_ocr(words_bbox_by_line, entities_list)
    elif model_ner == 'spacy':
        entities = get_entity_bboxes(words_bbox_by_line, base_spacy_model, my_spacy_model)

    # Корректировка NER VLLM
    if correct_ner and model_ner != 'qwen':
        entities_corrected_str = correct_ner_llm(model_vl, tokenizer_instruct, text_by_lines, entities)
        entities_corrected = ast.literal_eval(entities_corrected_str)
        entities = filter_entities_by_score(entities, entities_corrected, threshold=0.8)

    print(entities)
        
    ner_image = ner_draw(image.copy(), image_path, entities)

    redacted_image = mask_image(ner_image.copy(), image_path, entities)

    return ocr_image, ner_image, redacted_image

# image_path = "example/test_image.jpg"
# image = Image.open(image_path)
# ocr_image, ner_image, redacted_image = main(image, image_path, model_ner='spacy', correct_ocr=True, correct_ner=True)
