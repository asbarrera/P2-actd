import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Slider(
        id='mi-slicer',
        min=0,
        max=4,
        step=1,
        value=0
    ),
    # Aquí agregaría el resto de los componentes de su tablero
])

if __name__ == '__main__':
    app.run_server(debug=True)
