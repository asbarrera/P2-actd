import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output, State
import numpy as np
import os
import csv 
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.readwrite import BIFWriter
from pgmpy.readwrite import BIFReader
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.readwrite import XMLBIFWriter
from pgmpy.factors.discrete import TabularCPD

from pgmpy.readwrite import XMLBIFReader

#Datos normales
a="C:\\Users\\asbar\\OneDrive - Universidad de los Andes\\Carrera\\Séptimo semestre 2023-1\\ANALITICA\\P2\\processed.cleveland_repaired.csv"

data = pd.read_csv(a)

dict_df = data.to_dict('list')

#Datos modificados


      

b="C:\\Users\\asbar\\OneDrive - Universidad de los Andes\\Carrera\\Séptimo semestre 2023-1\\ANALITICA\\P2\\heart_disease_modified2.csv"
df = pd.read_csv(b)

df['sex'] = df['sex'].astype(float)
df['cp'] = df['cp'].astype(float)
df['fbs'] = df['fbs'].astype(float)
df['thalach'] = df['thalach'].astype(float)
df['exang'] = df['exang'].astype(float)
df['num'] = df['num'].astype(float)
df['anom_thalach'] = df['anom_thalach'].astype(float)


model = BayesianNetwork([('age','chol'),('age','fbs'),('sex','chol'),('sex','fbs')
                      ,('chol','num'),('fbs','num')
                      , ('num','cp'), ('num','exang'), ('num','anom_thalach'), ('num','trestbps')])



model.fit(df, estimator=MaximumLikelihoodEstimator)
#serializar
writer = BIFWriter(model)
writer.write_bif(filename='C:\\Users\\asbar\\OneDrive - Universidad de los Andes\\Carrera\\Séptimo semestre 2023-1\\ANALITICA\\P2\\proyectofinal.bif')

#deserializar
reader = BIFReader("C:\\Users\\asbar\\OneDrive - Universidad de los Andes\\Carrera\\Séptimo semestre 2023-1\\ANALITICA\\P2\\proyectofinal.bif")
modelo = reader.get_model()


# Crea la aplicación--------------------------------------------------------------------------------------------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])
image_url = "https://uniandes.edu.co/sites/default/files/logo-uniandes.png"
creador="https://img.freepik.com/vector-premium/icono-perfil-avatar_188544-4755.jpg?w=2000"

# Define el layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.Img(src=image_url, style={'position': 'absolute', 'bottom': 0, 'left': '0px','height': '78px'}),
        dcc.Link('Inicio ⌂', href='/',
                 style={'display': 'block','color': '#566573'}),
        dcc.Link('Probabilidad ⚄', href='/servicio-1',
                 style={'display': 'block','color': '#566573'}),
        dcc.Link('Corazón sano ❤', href='/servicio-2',
                 style={'display': 'block','color': '#566573'}),
        dcc.Link('Creadores', href='/servicio-3',
                 style={'display': 'block','color': '#566573'}),
    ], className='menu', 
    style={
        'position': 'fixed',
        'left': 0,
        'top': 0,
        'bottom': 0,
        'width': '140px',
        'padding': '10px',
        'background-color': '#f8f8f8',
        'border-right': '1px solid #ddd'
    }
    ),
    
    # Define el contenido principal
    html.Div(id='contenido')
])

# Paginas principal-----------------------------------------------------------------------------------------------------------------------------
#Graficos
# histograma

# Calcular la mediana del colesterol
chol_median = np.median(dict_df['chol'])

# Crear el histograma
histograma = dcc.Graph(
    id='histograma',
    figure={
        'data': [{
            'x': dict_df['chol'],
            'type': 'histogram'
        }],
        'layout': {
            'title': 'Distribución del colesterol',
            'xaxis': {'title': 'Colesterol mg/dl'},
            'yaxis': {'title': 'Frecuencia'}
        }
    }
)

# Gráfico de dispersión
corr_coef = np.corrcoef(dict_df['age'], dict_df['thalach'])[0, 1]

dispersion = dcc.Graph(
    id='dispersion',
    figure={
        'data': [{
            'x': dict_df['age'],
            'y': dict_df['thalach'],
            'mode': 'markers'
        }],
        'layout': {
            'title': 'Frecuencia cardiaca dado la edad',
            'xaxis': {'title': 'Edad'},
            'yaxis': {'title': 'Frecuencia cardiaca'},
            'annotations': [{
                'x': 0.95,
                'y': 0.95,
                'showarrow': False,
                'text': 'Coeficiente de correlación: {:.2f}'.format(corr_coef),
                'xref': 'paper',
                'yref': 'paper',
                'align': 'right',
                'font': {
                    'size': 10
                }
            }]
        }
    }
)
#Grafico de torta
fbs= data['fbs'].tolist()
cuenta = [fbs.count(0), fbs.count(1)]
labels = ['fbs < 120 mg/dl', 'fbs > 120 mg/dl']
colors = ['#3498DB', '#AED6F1 ']
trace = go.Pie(labels=labels, values=cuenta, marker=dict(colors=colors))
fig = go.Figure(data=[trace])
grafico_torta = dcc.Graph(
    id='mi_grafico',
    figure={
        'data': [trace],
        'layout': {
            'title': 'Personas con un fbs superior a 120 mg/dl ',
            'height': 400,
            'showlegend': True,
            'legend': {'x': 1, 'y': 0.5},
            
        },
    }
)

#Layout

home = html.Div([
    html.H1('¡Bienvenido/a a nuestra página de prevención de enfermedades cardiovasculares!', style={'backgroundColor': '#34495e', 'color': 'white','padding': '20px'}),
    html.Div([
        html.P('Nos complace que haya decidido visitar nuestro sitio web para obtener información y consejos sobre cómo prevenir enfermedades cardiovasculares y mantener su corazón sano.Sabemos que la salud del corazón es esencial para llevar una vida plena y activa, y estamos comprometidos a brindarle toda la información necesaria para ayudarlo/a a lograrlo.')
    ]),
    html.H4('Algunos gráficos que le pueden interesar'),
    # Columna izquierda
    dbc.Row([
        dbc.Col(histograma, width={'size': 4}),
        dbc.Col(dispersion, width={'size': 4}),
        dbc.Col(grafico_torta, width={'size': 4})
    ]),
    html.Div([
        html.P('Estos datos corresponden a una muestra de 606 pacientes proporcionada por la Universidad de California y están relacionados con enfermedades cardiovasculares'),
    ]),
    
    html.Div([
        html.H5('En nuestra página puede a partir de unos datos calcular la probabilidad de tener una enfermedad cardiovascular o ver buenos hábitos para poder prevenir las enfermedades cardiovasculares'),
    ])

],style={'marginLeft': '132px', 'padding': '20px'})

# Pagina probabilidad------------------------------------------------------------------------------------------------------------------------------------

options_sexo= [
    {'label': 'Hombre', 'value': 'Hombre'},
    {'label': 'Mujer', 'value': 'Mujer'},
    ]

options_col=[
    {'label':'0-240','value':'0-240'},
    {'label':'240-260','value':'240-260'},
    {'label':'260-400','value':'260-400'}
    ]


servicio_1 = html.Div([
    html.H1('Probabilidad de tener una enfermedad cardiovascular', style={'backgroundColor': '#34495e', 'color': 'white','padding': '20px'}),
    html.Div([
            html.P('Para calcular la probabilidad necesitamos que tenga disponible la siguiente información:')
            ]),
            html.Div([
            html.P("1: Un examen que muestre su nivel actual de colesterol mg/dl")
            ]),
            html.Div([
            html.P("2: Un examen que muestre su nivel actual de glucemia en ayunas (o fasting blood sugar) mg/dl")
            ]),
            html.Div([
            html.P("3: Un examen que muestre su frecuencia cardiaca en reposo (se puede medir tomando el pulso en la muñeca o el cuello durante un período de tiempo determinado, generalmente de 30 segundos a un minuto. El resultado se multiplica luego por dos para obtener la frecuencia cardíaca en reposo) ")
            ]),
            
         html.Div([
         html.H5('Seleccione su sexo.')]),
           dcc.Dropdown(
            id='dropdown',
            options=options_sexo,
            value='Hombre',
            style={'display': 'inline-block', 'width': '300px'}
           ),
           html.Div(id='output'),
           html.Br(),
           html.H5("¿Qué edad tiene?"),
           html.Div(["Edad: ",
              dcc.Input(id='my-input', 
                        value='0', 
                        type='text',
                        style={
                      'display': 'inline-block', 
                      'width': '300px', 
                      'border': '1px solid gray'
                  })]),

             html.Div(id='output'),
             html.Br(),
    html.Div([
      html.H5('Seleccione su nivel de colesterol mg/dl.')]),
    dcc.Dropdown(
        id='dropdown_col',
        options=options_col,
        value='',
        style={'display': 'inline-block', 'width': '300px'}
        ) ,
        
    html.Div(id='output'),
    html.Br(),
    html.H5("Escriba su nivel de glucemia en ayunas mg/dl"),
    html.Div(["Fbs: ",
              dcc.Input
              (id='input-fbs', value='0', type='text',
               style={
                      'display': 'inline-block', 
                      'width': '300px', 
                      'border': '1px solid gray'
                  })])  ,
    html.Br(),
    html.Div(id='output'),
    html.H5("Escriba su frecuencia cardiaca"),
    html.Div(["Frecuencia cardiaca: ",
              dcc.Input
              (id='input-car', value='0', type='text',
               style={
                      'display': 'inline-block', 
                      'width': '300px', 
                      'border': '1px solid gray'
                  })]),
    html.Br(),
    html.Button('Enviar', id='button', style={'color': '#273746','background-color': '#D4E6F1'}),
        html.Br(),
        html.Div(id='output'
                  )   


],
style={'marginLeft': '132px', 'padding': '20px'})

#Página de buenos hábitos-----------------------------------------------------------------------------------------------------------------------------------------
servicio_2 = html.Div([
    html.H1('Buenos hábitos para el corazón', style={'backgroundColor': '#34495e', 'color': 'white','padding': '20px'}),
    html.P('No hay que tomarse una enfermedad cardiovascular a la ligera, por eso aca le damos algunas recomendaciones para evitarlas y mantener su corazón sano.'),
    html.Div([
        html.P("1. Mantener una dieta saludable: Una dieta equilibrada y saludable puede ayudar a reducir el riesgo de enfermedades cardiovasculares. Se recomienda una dieta rica en frutas, verduras, cereales integrales, proteínas magras y grasas saludables.")
    ]),
html.Div([
html.P("2. Mantenerse activo: El ejercicio regular es una excelente manera de mantener un corazón saludable. Se recomienda al menos 30 minutos de actividad física moderada la mayoría de los días de la semana")
    ]),
html.Div([
html.P("3. No fumar: Fumar es un factor de riesgo importante para las enfermedades cardiovasculares. Dejar de fumar puede reducir significativamente este riesgo")
    ])
], style={'marginLeft': '150px', 'padding': '20px'})

#Página de creadores-----------------------------------------------------------------------------------------------------------------------------------------------

image_url = "https://img.freepik.com/vector-premium/icono-perfil-avatar_188544-4755.jpg?w=2000"

description_1 = html.Ul([
    html.Li("Adriana Sofia Barrera"),
    html.Li("202011470")
])


description_2 = html.Ul([
    html.Li("Juan José Flórez"),
    html.Li("202011038")
])


description_3 = html.Ul([
    html.Li("Juan Sebastián Hernández"),
    html.Li("201715595")
])

image_list = html.Div([
    html.Div([
        html.Img(src=image_url, height="145px"),
        html.Div(description_1, style={"display": "inline-block", "vertical-align": "top",'fontSize': 20})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    
    html.Div([
        html.Img(src=image_url, height="145px"),
        html.Div(description_2, style={"display": "inline-block", "vertical-align": "top",'fontSize': 20})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    
    html.Div([
        html.Img(src=image_url, height="145px"),
        html.Div(description_3, style={"display": "inline-block", "vertical-align": "top",'fontSize': 20})
    ], style={'display': 'flex', 'flex-direction': 'row'})
])


servicio_3 = html.Div([
    html.H1('Creadores',style={'backgroundColor': '#34495e', 'color': 'white','padding': '20px'}),
    image_list
], style={'marginLeft': '150px', 'padding': '20px'})



# Define el callback para cambiar el contenido del Div de contenido-------------------------------------------------------------------------------------------------
@app.callback(
    Output('contenido', 'children'),
    Input('url', 'pathname')
)
def cambiar_contenido(pathname):
    if pathname == '/':
        return home
        
    elif pathname == '/servicio-1':
        return servicio_1
    
    elif pathname == '/servicio-2':
        return servicio_2
    
    elif pathname == '/servicio-3':
        return servicio_3
    
    else:
        return home

# Calculo de probabilidad---------------------------------------------------------------------------------------------------------------------------------------------------------
sexo=""
edad=""
col =""
fbs=""
proba=0
anom_thalach=0
cp=""
recomendacion=""
@app.callback(
    Output('output', 'children'),
    Input('url', 'pathname'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('dropdown', 'value')],
    [dash.dependencies.State('my-input', 'value')],
    [dash.dependencies.State('dropdown_col', 'value')],
    [dash.dependencies.State('input-fbs', 'value')],
    [dash.dependencies.State('input-car', 'value')]
)


def update_output(pathname, n_clicks, dropdow_value,input_value,dropdown_col,input_fbs,input_car):
    if pathname== '/servicio-1':
        if n_clicks:
            global sexo
            sexo = dropdow_value
            if sexo=="Hombre":
                sexo=1
            else:
                sexo=0
                
            global edad
            edad= input_value
            if int(edad)<=44:
                edad="0-45"
            elif int(edad)<=54:
                edad="45-55"
            else:
                edad='55-100'
            global col
            col= dropdown_col
            global fbs
            fbs= input_fbs
            if int(fbs)>120:
                fbs=1
            else:
                fbs=0
            
            global anom_thalach
            anom_thalach=input_car
            if int(anom_thalach)>(220-int(input_value)):
                anom_thalach=1
            else:
                anom_thalach=0
            
            global proba
            global recomendacion
            infer = VariableElimination(model)
            P1 = infer.query(['num'], evidence={'age': edad,'sex':sexo, 'chol':col,'fbs':fbs,"anom_thalach":anom_thalach})
            proba= P1.values

            if proba[0]>0.3:
                recomendacion="Tienes una probabilidad alta, deberias ir al médico para revisar cómo esta tu corazón."
            else:
                recomendacion="Tienes una probabilidad baja, lo que significa que tu corazón esta sano."

            return html.Div([
                html.H4("Tu probabilidad de tener una enfermedad cardiovascular es de :" +str(round(proba[0]*100,1))+ "%"),
                html.P(recomendacion)
                             ])
        
        else:
            return ''
    
    return dash.no_update




# Ejecuta la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)