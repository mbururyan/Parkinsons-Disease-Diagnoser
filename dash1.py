# Import Dash Libraries
import dash
from dash import html
from dash import dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import researchpy as rp

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

df = pd.read_csv('/Users/RyanMburu/Documents/DS-Projects/Supervised-Learning/Parkinsons/Clean Parkinsons (1).csv')

# Picture link
pic1 = '/Users/RyanMburu/Documents/DS-Projects/Supervised-Learning/Parkinsons/hosi1.png'

list_agile = list(df['Leg agility'].unique())

# Navigation Bar
navbar = dbc.Navbar(
    [
        html.A(
            dbc.Row(
                [
                    dbc.Col(html.Img(src=pic1, height='30px')),
                    dbc.Col(dbc.NavbarBrand('PARKINSONS DISEASE DETECTOR', className='ml-2')),
                ], style={'margin-left' : '11%'},
                align = 'center',
                no_gutters=True,
            ),
        )
    ], color='#A3D8FF',
    dark=True,
    sticky='top',
    style={'width': 'calc(100% - 12rem)', 'float': 'right', 'height': '4.5rem'}
)

# Sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "15rem",
    "padding": "2rem 1rem",
    "background-color": '#BFCDD9',
}

CONTENT_STYLE = {
    'margin-left' : '18rem',
    'margin-right' : '2rem',
    'padding' : '2rem 2 rem',
}

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            'What Page do you want to view ?'
        ),
        dbc.Nav(
            [
                dbc.NavLink('HomePage', href='/', active='extact'),
                dbc.NavLink('Neurological Details', href='/page-1', active='extact'),
            ],
            pills=True,
            vertical=True,
        ),
    ],style=SIDEBAR_STYLE,
)


# Information cards
card_age = dbc.Card(
    [
        dbc.CardHeader('Most Popular Age group suffering From Parkinsons is :'),
        dbc.CardBody(
            [
                html.P('73')
            ]
        )
    ], color='#A3D8FF',
)

card_gender = dbc.Card(
    [
        dbc.CardHeader('Most Popular Gender suffering From Parkinsons is :'),
        dbc.CardBody(
            [
                html.P('Male')
            ]
        )
    ], color='#A3D8FF',
)

card_med = dbc.Card(
    [
        dbc.CardHeader('Most Popular Benzodiazepine Medication is :'),
        dbc.CardBody(
            [
                html.P('Rivotril')
            ]
        )
    ], color='#A3D8FF',
)


# Deck for the cards
cards1 = dbc.CardDeck(
    [
        dbc.Col(card_age, width='auto'),
        dbc.Col(card_gender, width='auto'),
    ],
)
# User Inputs

# Dropdowns
agile_dropdown = dbc.Card(
    [
        html.Div(
            [
                dbc.Label('Is the patients leg agile ? '),
                dcc.Dropdown(
                    id = 'age_dropdown',
                    options = [{'label' : i, 'value':i} for i in list_agile],
                    value='Age'
                ),
            ]
        )
    ], body=True, color='dark', outline=True
)

# Age

age_input = dbc.Card(
    [
        html.Div(
            [
                dbc.Label('Whats the patients age ? : '),
                dcc.Input(
                    id = 'age_input',
                    placeholder = 'Age : ',
                    type = 'number',
                ),
                html.Br(),
            ]
        )
    ], body=True, color='dark', outline=True
)

# Tremors
list_tremors = list(df['Tremor'].unique())

tremor_dropdown = dbc.Card(
    [
        html.Div(
            [
                dbc.Label('Is the patient experiencing tremors ? : '),
                dcc.Dropdown(
                    id = 'tremor_dropdown',
                    options = [{'label' : i, 'value':i} for i in list_tremors],
                    value='Tremor'
                ),
            ]
        )
    ], body=True, color='dark', outline=True
)

# History in the Family
list_history = list(df['History'].unique())

history_dropdown = dbc.Card(
    [
        html.Div(
            [
                dbc.Label('History in the family? : '),
                dcc.Dropdown(
                    id = 'history_dropdown',
                    options = [{'label' : i, 'value':i} for i in list_history],
                    value='History'
                ),
            ]
        )
    ], body=True, color='dark', outline=True
)

# Gender
list_gender = list(df['Gender'].unique())

gender_dropdown = dbc.Card(
    [
        html.Div(
            [
                dbc.Label('Gender of the patient : '),
                dcc.Dropdown(
                    id = 'gender_dropdown',
                    options = [{'label' : i, 'value':i} for i in list_gender],
                    value='Gender'
                ),
            ]
        )
    ], body=True, color='dark', outline=True
)

# Antidepressants
list_anti = list(df['Antidepressant therapy'].unique())

anti_dropdown = dbc.Card(
    [
        html.Div(
            [
                dbc.Label('Is patient on antidepressants ? : '),
                dcc.Dropdown(
                    id = 'anti_dropdown',
                    options = [{'label' : i, 'value':i} for i in list_anti],
                    value='Anti'
                ),
            ]
        )
    ], body=True, color='dark', outline=True
)

# Benzodiazepine meds
list_benz = list(df['Benzodiazepine medication'].unique())

benz_dropdown = dbc.Card(
    [
        html.Div(
            [
                dbc.Label('Is patient on other meds ? : '),
                dcc.Dropdown(
                    id = 'benz_dropdown',
                    options = [{'label' : i, 'value':i} for i in list_benz],
                    value='Meds'
                ),
            ]
        )
    ], body=True, color='dark', outline=True
)

#Posture Performance
list_posture = list(df['Posture'].unique())

posture_dropdown = dbc.Card(
    [
        html.Div(
            [
                dbc.Label('How is the patients posture rating ? : '),
                dcc.Dropdown(
                    id = 'posture_dropdown',
                    options = [{'label' : i, 'value':i} for i in list_posture],
                    value='Rating'
                ),
            ]
        )
    ], body=True, color='dark', outline=True
)

# Deck with info input on age, gender, history in fam and presence of tremors

cards2 = dbc.CardDeck(
    [
        dbc.Col(age_input, width='auto'),
        dbc.Col(gender_dropdown, width='auto'),
        dbc.Col(tremor_dropdown, width='auto'),
        dbc.Col(history_dropdown, width='auto'),
    ],
)

cards3 = dbc.CardDeck(
    [
        dbc.Col(agile_dropdown, width='auto'),
        dbc.Col(anti_dropdown, width='auto'),
        dbc.Col(benz_dropdown, width='auto'),
        dbc.Col(posture_dropdown, width='auto')
    ]
)

# App layout
app.layout = dbc.Container(
    # Title
    [
        dcc.Location(id='url'),
        sidebar,
        html.H1(
            'PARKINSONS DISEASE PREDICTOR',
        ),
        html.Hr(),

        #Info cards
        dbc.Row(
            [
                html.Div([cards1])
            ]
        ),
        html.Hr(),

        html.H1(
            'Patients General Information '
        ),
        html.Hr(),

        #DropDowns first row
        dbc.Row(
            [
                html.Div([cards2])
            ]
        ),
        html.Hr(),

        #Dropdowns second row
        dbc.Row(
            [
                html.Div([cards3])
            ]
        ),
        html.Hr(),

        html.Div(
            children=[
                html.Label('Leg Agility'),
                dcc.RadioItems(
                    id='agility',
                    options = [{'label' : i, 'value':i} for i in list_agile]

                )
            ]
        )



    ], fluid=True,
    style=CONTENT_STYLE

)

# Sidebar Callback



# Call the app
if __name__ == '__main__':
    app.run_server(debug=True)