from fastapi.middleware.wsgi import WSGIMiddleware
from dash import html, dcc, Input, Output, State
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from pii_module import main
from PIL import Image
import base64
import dash
import time
import io


fastapi_app = FastAPI()


dash_app = dash.Dash(__name__, requests_pathname_prefix="/dashboard/")
server = dash_app.server


def decode_base64_image(base64_string):

    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    
    return Image.open(io.BytesIO(img_data))


def block_with_image(title, image_pil, filename):
    return html.Div([
        html.H4(title, style={'marginBottom': '10px'}),
        html.Img(src=image_pil, style={'width': '100%', 'borderRadius': '10px', 'boxShadow': '0 4px 10px rgba(0, 0, 0, 0.1)'}),
        html.A("Скачать", href=image_pil, download=filename, target="_blank",
               style={
                   'display': 'inline-block',
                   'marginTop': '10px',
                   'padding': '8px 14px',
                   'backgroundColor': '#1976D2',
                   'color': 'white',
                   'textDecoration': 'none',
                   'borderRadius': '5px',
                   'boxShadow': '0 2px 5px rgba(0,0,0,0.2)'
               })
    ], style={
        'margin': '20px auto',
        'padding': '20px',
        'width': '60%',
        'backgroundColor': '#FAFAFA',
        'borderRadius': '12px',
        'boxShadow': '0 4px 10px rgba(0,0,0,0.05)',
        'textAlign': 'center'
    })

dash_app.title = "Masking Service"


dash_app.layout = html.Div([
    html.Div([
        html.H2("Сервис маскирования персональных данных", style={
            'textAlign': 'center',
            'marginBottom': '30px',
            'color': '#333',
            'fontWeight': 'bold'
        }),

        dcc.Upload(
            id='upload-image',
            children=html.Button("Загрузить изображение", style={
                'padding': '15px 25px',
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontSize': '16px'
            }),
            style={
                'width': '50%',
                'margin': '0 auto',
                'textAlign': 'center',
                'marginBottom': '20px'
            },
            multiple=False
        ),

        html.Div([
            html.Label("Модель NER:", style={
                'fontWeight': 'bold',
                'marginRight': '10px'
            }),
            dcc.Dropdown(
                id='ner-model-select',
                options=[
                    {'label': 'spaCy', 'value': 'spacy'},
                    {'label': 'Qwen', 'value': 'qwen'}
                ],
                value='qwen',
                clearable=False,
                style={
                    'width': '160px',
                    'marginRight': '30px',
                    'display': 'inline-block'
                }
            ),
            html.Div([
                dcc.Checklist(
                    id='correction-options',
                    options=[
                        {'label': 'Корректировать OCR', 'value': 'correct_ocr'},
                        {'label': 'Корректировать NER', 'value': 'correct_ner'}
                    ],
                    value=['correct_ocr', 'correct_ner'],
                    inputStyle={"marginRight": "6px", "marginLeft": "10px"},
                    labelStyle={'display': 'inline-block', 'marginRight': '20px'}
                )
            ], style={'display': 'inline-block'})
        ], style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'gap': '20px',
            'marginBottom': '30px',
            'flexWrap': 'wrap'
        }),

        html.Div(id='loading-status', style={'textAlign': 'center', 'color': '#555'}),
        html.Div(id='output-images', style={'marginTop': '30px'})
    ], style={
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '40px',
        'backgroundColor': '#f9f9f9',
        'borderRadius': '15px',
        'boxShadow': '0px 0px 20px rgba(0, 0, 0, 0.1)'
    })
], style={
    'fontFamily': 'Arial, sans-serif',
    'backgroundColor': '#f0f2f5',
    'minHeight': '100vh',
    'paddingTop': '30px'
})

@dash_app.callback(
    Output('output-images', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('ner-model-select', 'value'),
    State('correction-options', 'value')
)
def update_output(content, filename, selected_model, correction_options):
    if content is None:
        return html.Div()

    try:
        pil_img = decode_base64_image(content)

        correct_ocr = 'correct_ocr' in correction_options
        correct_ner = 'correct_ner' in correction_options

        start_time = time.time()

        ocr_image, ner_image, redacted_image = main(
            pil_img,
            filename,
            model_ner=selected_model,
            correct_ocr=correct_ocr,
            correct_ner=correct_ner
        )

        end_time = time.time()
        elapsed = round(end_time - start_time, 2)

        return html.Div([
            html.H4("Результаты обработки", style={'textAlign': 'center'}),
            html.Div([
                html.P(f"Время обработки: {elapsed} сек", style={
                    'textAlign': 'center',
                    'color': '#888',
                    'fontStyle': 'italic',
                    'marginTop': '-10px'
                }),
                block_with_image("OCR:", ocr_image, "ocr_result.png"),
                block_with_image("NER:", ner_image, "ner_result.png"),
                block_with_image("PII Masking:", redacted_image, "redacted_result.png")
            ])
        ])
    except Exception as e:
        return html.Div([html.H4("Ошибка обработки:"), html.Pre(str(e))])

fastapi_app.mount("/dashboard", WSGIMiddleware(dash_app.server))

@fastapi_app.get("/api/hello")
def hello():
    return {"message": "Hello from FastAPI"}

@fastapi_app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <meta http-equiv="refresh" content="0; URL='/dash'" />
        </head>
        <body>
            <p>Перенаправление на интерфейс Dash...</p>
        </body>
    </html>
    """

# uvicorn app:fastapi_app --reload
# http://127.0.0.1:8000/dashboard/