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

# ML section
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

df = pd.read_csv('/Users/RyanMburu/Documents/DS-Projects/Supervised-Learning/Parkinsons/Clean_dash.csv')

# ML Model loading and preparation
final_model = load_model('final_model.h5')

# Splitting
X = df.loc[:, 'Age' : 'Latency of\nrespiratory exchange (ms).1'].values
y = df['Status'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scaling using MinMax


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
                dbc.NavLink('Parkinsons Diagnosis', href='/page-1', active='extact'),
            ],
            pills=True,
            vertical=True,
        ),
        html.Hr(),
    ],style=SIDEBAR_STYLE,
)

# Form for user input

# Age 

age_input = dbc.Row(
    [
        dbc.Label('Patients Age : ', html_for='example-age-row', width=4),
        dbc.Col(
            dbc.Input(
                type='number', id='Age', placeholder='Enter Patients Age : '
            ),
            width=8,
        ),
    ], className='mb-3',
)

# History in the fam
list_history = list(df['History'].unique())
history_input = dbc.Row(
    [
        dbc.Label('Family History : ', html_for='example-radios-row', width=4),
        dbc.Col(
            dbc.RadioItems(
                id='History', 
                options= [{'label' : i, 'value':i} for i in list_history],
            ),
            width=8,
        ),
    ], className='mb-3',
)

# Tremor
list_tremor = list(df['Tremor'].unique())
tremor_input = dbc.Row(
    [
        dbc.Label('Experiencing Tremors? ', html_for='example-radios-row', width=4),
        dbc.Col(
            dbc.RadioItems(
                id='Tremor', 
                options= [{'label' : i, 'value':i} for i in list_tremor],
            ),
            width=8,
        ),
    ],className='mb-3',
)

# Leg Agility
list_agility = list(df['Leg agility'].unique())
agility_input = dbc.Row(
    [
        dbc.Label('Is patients leg agile ?', html_for='example-radios-row', width=4),
        dbc.Col(
            dbc.RadioItems(
                id='Leg_agility', 
                options= [{'label' : i, 'value':i} for i in list_agility],
            ),
            width=8,
        ),
    ],className='mb-3',
)


# Entropy of speech timing (-)
entropy_input = dbc.Row(
    [
        dbc.Label('Patients Entropy of Speech : ', html_for='example-age-row', width=4),
        dbc.Col(
            dbc.Input(
                type='number', id='Entropy_of_speech_timing', placeholder='Range of 1 - 2: '
            ),
            width=8,
        ),
    ], className='mb-3',
)

# Acceleration of speech timing(-/min2)
acceleration_input = dbc.Row(
    [
        dbc.Label(' Patients acceleration of speech timing : ', html_for='example-age-row', width=4),
        dbc.Col(
            dbc.Input(
                type='number', id='Acceleration_of_speech_timing', placeholder='Enter patients Acceleration of speech timing in (-/min2) : '
            ),
            width=8,
        ),
    ], className='mb-3',
)

# Duration of pause intervals (ms)
pause_intervals_input = dbc.Row(
    [
        dbc.Label(' Patients duration of pause intervals : ', html_for='example-age-row', width=4),
        dbc.Col(
            dbc.Input(
                type='number', id='Duration_of_pause_intervals', placeholder='Pause intervals in minutes per sec : '
            ),
            width=8,
        ),
    ], className='mb-3',
)

# Gapingin-between voiced\nintervals(-/min)'
gaping_intervals_input = dbc.Row(
    [
        dbc.Label(' Patients duration between intervals : ', html_for='example-age-row', width=4),
        dbc.Col(
            dbc.Input(
                type='number', id='Gaping_between_voiced_intervals', placeholder='Intervals in -/min : '
            ),
            width=8,
        ),
    ], className='mb-3',
)

# Duration of unvoiced stops (ms)
unvoiced_stops_input = dbc.Row(
    [
        dbc.Label(' Patients duration of unvoiced stops : ', html_for='example-age-row', width=4),
        dbc.Col(
            dbc.Input(
                type='number', id='Duration_of_unvoiced_stops', placeholder='Duration in milliseconds : '
            ),
            width=8,
        ),
    ], className='mb-3',
)

# Decay of unvoiced fricatives(promile/min)
fricatives_input = dbc.Row(
    [
        dbc.Label(' Patients unvoiced fricatives : ', html_for='example-age-row', width=4),
        dbc.Col(
            dbc.Input(
                type='number', id='Decay_of_unvoiced_fricatives', placeholder='In promile / min : '
            ),
            width=8,
        ),
    ], className='mb-3',
)


# Entropy of speech timing (-).1
entropy_speech_input = dbc.Row(
    [
        dbc.Label(' Patients entropy of speech input : ', html_for='example-age-row', width=4),
        dbc.Col(
            dbc.Input(
                type='number', id='Entropy_of_speech_input', placeholder='Entropy of speech timing: '
            ),
            width=8,
        ),
    ], className='mb-3',
)


# Acceleration of speech timing (-/min2)
acceleration_speech_input = dbc.Row(
    [
        dbc.Label(' Patients acceleration of speech input : ', html_for='example-age-row', width=4),
        dbc.Col(
            dbc.Input(
                type='number', id='Acceleration_of_speech_input', placeholder='Acceleration of speech timing: '
            ),
            width=8,
        ),
    ], className='mb-3',
)

# Duration of unvoiced stops (ms).1
duration_unvoiced_intervals_input = dbc.Row(
    [
        dbc.Label(' Patients duration between unvoiced intervals : ', html_for='example-age-row', width=4),
        dbc.Col(
            dbc.Input(
                type='number', id='Duration_of_unvoiced_intervals', placeholder='Duration in millisecondss : '
            ),
            width=8,
        ),
    ], className='mb-3',
)


# Rate of speech respiration (- /min)
rate_speech_respiration_input = dbc.Row(
    [
        dbc.Label(' Patients rate of speech respiration : ', html_for='example-age-row', width=4),
        dbc.Col(
            dbc.Input(
                type='number', id='Rate_of_speech_respiration', placeholder='Rate of the patients speech in milliseconds : '
            ),
            width=8,
        ),
    ], className='mb-3',
)

# Latency of\nrespiratory exchange (ms).1
latency_respiration_exchange_input = dbc.Row(
    [
        dbc.Label(' Latency of respiratory exchange : ', html_for='example-age-row', width=4),
        dbc.Col(
            dbc.Input(
                type='number', id='Latency_of_respiratory_exchange', placeholder='Rate of the patients respiration in milliseconds : '
            ),
            width=8,
        ),
    ], className='mb-3',
)


# Components in a form array
form = dbc.Form([age_input, history_input, tremor_input, agility_input])

form2 = dbc.Form([entropy_input, acceleration_input, pause_intervals_input, gaping_intervals_input, unvoiced_stops_input, fricatives_input, entropy_speech_input,
acceleration_speech_input, duration_unvoiced_intervals_input, rate_speech_respiration_input, latency_respiration_exchange_input])

# Button
button = html.Div(
    [
        dbc.Button('Predict', color='success', id='predict_button'),
        html.Span(id = 'predict_output', style={"verticalAlign": "middle"}),
    ],  className = 'd-grid gap-2 col-6 mx-auto'
)

# Information cards
card_age = dbc.Card(
    [
        dbc.CardHeader('Popular Age grp affected :'),
        dbc.CardBody(
            [
                html.P('73')
            ]
        )
    ], color='#DEDEDE',
)

card_gender = dbc.Card(
    [
        dbc.CardHeader('Popular Gender:'),
        dbc.CardBody(
            [
                html.P('Male')
            ]
        )
    ], color='#DEDEDE',
)

card_med = dbc.Card(
    [
        dbc.CardHeader('Popular Medication is :'),
        dbc.CardBody(
            [
                html.P('Rivotril')
            ]
        )
    ], color='#DEDEDE',
)

card_dawa = dbc.Card(
    [
        dbc.CardHeader('Is there a cure:'),
        dbc.CardBody(
            [
                html.P('No Cure')
            ]
        )
    ], color='#DEDEDE',
)

# Deck for the cards
cards1 = dbc.CardDeck(
    [
        dbc.Col(card_age, width='auto'),
        dbc.Col(card_gender, width='auto'),
        dbc.Col(card_med, width='auto'),
        dbc.Col(card_dawa, width='auto')
    ],
)

# Barchart DropDown
df_yes = df[df['Status'] == 1] 
plot_df = df_yes.loc[:, 'Age' : 'Leg agility']
age_gender = list(plot_df)

bar_dropdown = dbc.Card(
    [
        html.Div(
            [
                dbc.Label('Which characteristics of patients do you want to vizualize?'),
                dcc.Dropdown(
                    id='bar_dropdown',
                    #set the options after loading dataframe
                    options=[{'label':i, 'value':i} for i in age_gender],
                    value='Age',
                ),
            ]
        )
    ], body=True
)


# Website layout

content = html.Div(
    id = 'HomePage',
    children=[],
    style = CONTENT_STYLE
)

app.layout = html.Div(
    [
        dcc.Location(id='url'),
        sidebar,
        content,
    ]
)


# Sidebar Callback
@app.callback(
    Output('HomePage', 'children'),
    [
        Input('url', 'pathname')
    ]
)

def page_switcher(pathname):
    #Home page
    if pathname == '/':
        return[
            html.H1('PARKINSONS DISEASE HOME PAGE',
            style={'text-align' : 'center'}),
            html.Hr(),
            html.H2('Information on the Disease'),
            html.Hr(),           
            html.Div(
                [
                    cards1
                ]
            ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(bar_dropdown, md=3, width='auto'),
                    dbc.Col(dcc.Graph(id='bar_graph'), width='auto'),
                    html.Hr()
                ]
            )
        ]

    # Questions page
    elif pathname == '/page-1':
        return [
            html.H1('PARKINSONS DISEASE PREDICTOR',
            style={'text-align' : 'center'}),
            html.Hr(),
            html.H2('Patients General Information'),
            html.P('When choosing values, 1 - > yes and 0 - > No'),
            html.Hr(),
            html.Div(
                [
                    form
                ]
            ),
            html.Hr(),
            html.H2('Patients Neurological Information after taking measurements'),
            html.Hr(),
            html.Div(
                [
                    form2
                ]
            ),
            html.Hr(),
            html.H3('Patients Description '),
            html.Hr(),
            html.Div(
                id='light_info',
            ),
            html.Hr(),
            html.H3('Prediction Outcome : '),
            html.Hr(),
            html.Div(
                id='result',
            ),
            html.Hr(),
            html.H3('Websites accuracy in Prediction : '), 
            html.Hr(),
            html.H6('98% Accuracy.')          

            # html.Hr(),
            # html.Div(
            #     [
            #         button
            #     ]
            # )
        ]

# Barchart callback
@app.callback(
    Output('bar_graph', 'figure'),
    Input('bar_dropdown', 'value')
)

def bar(column):
    df_new = plot_df[[column]].value_counts().reset_index()
    df_new.rename(columns={0 : 'Sum of patients'}, inplace=True)
    
    plot = px.bar(df_new, x=column, y='Sum of patients', color=column)
    return plot


# Form callback

# Patient info callback
@app.callback(
    Output('light_info', 'children'),
    Input('Age', 'value'),
)

def prediction(Age):
    return u'The age of the patient is {} and the website will predict if He/She has Perkinsons Disease ...'.format(Age)


@app.callback(
    Output('result', 'children'),
    Input('Age', 'value'),
    Input('History', 'value'),
    Input('Tremor', 'value'),
    Input('Leg_agility', 'value'),
    Input('Entropy_of_speech_timing', 'value'),
    Input('Acceleration_of_speech_timing', 'value'),
    Input('Duration_of_pause_intervals', 'value'),
    Input('Gaping_between_voiced_intervals', 'value'),
    Input('Duration_of_unvoiced_stops', 'value'),
    Input('Decay_of_unvoiced_fricatives', 'value'),
    Input('Entropy_of_speech_input', 'value'),
    Input('Acceleration_of_speech_input', 'value'),
    Input('Duration_of_unvoiced_intervals', 'value'),
    Input('Rate_of_speech_respiration', 'value'),
    Input('Latency_of_respiratory_exchange', 'value'),
)

def prediction(Age, History, Tremor, Leg_agility, Entropy_of_speech_timing, Acceleration_of_speech_timing, Duration_of_pause_intervals, Gaping_between_voiced_intervals, Duration_of_unvoiced_stops, Decay_of_unvoiced_fricatives, Entropy_of_speech_input, Acceleration_of_speech_input, Duration_of_unvoiced_intervals, Rate_of_speech_respiration, Latency_of_respiratory_exchange) :
    X_new = np.array([Age, History, Tremor, Leg_agility, Entropy_of_speech_timing, Acceleration_of_speech_timing, Duration_of_pause_intervals, Gaping_between_voiced_intervals, Duration_of_unvoiced_stops, Decay_of_unvoiced_fricatives, Entropy_of_speech_input, Acceleration_of_speech_input, Duration_of_unvoiced_intervals, Rate_of_speech_respiration, Latency_of_respiratory_exchange]).reshape(1, -1)
# scale the inputs
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_new = scaler.transform(X_new)

    # The last dance

    #Load the model
    Diagnosis=final_model.predict(X_new)

    # return u'The outcome is  {} '.format(Diagnosis)
    
    if Diagnosis < [1]:
        return [ u' Congratulations,Your patient does NOT have Perkinsons Disease !! ']
    elif Diagnosis > [1]:
        return [ u'Your patient has Perkinsons Disease. Please follow the necessary procedures to ensure the patient is cured. ']


# Call the app
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False)

