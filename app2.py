import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from data_simulator2 import generate_sensor_data, predict_failures, reset_fault_state
from utils import calculate_efficiency, calculate_vibration_amplitude
import pandas as pd
from datetime import datetime
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
import time

server = Flask(__name__)
server.secret_key = 'your-secret-key'

# Initialize Prometheus metrics
metrics = PrometheusMetrics(server)

# Custom Prometheus metrics for compressor monitoring
compressor_failure_probability = Gauge(
    'compressor_failure_probability',
    'Current failure probability of compressor',
    ['compressor_id', 'compressor_name']
)

compressor_sensor_reading = Gauge(
    'compressor_sensor_reading',
    'Current sensor readings from compressors',
    ['compressor_id', 'compressor_name', 'sensor_type', 'unit']
)

compressor_maintenance_alerts = Counter(
    'compressor_maintenance_alerts_total',
    'Total number of maintenance alerts generated',
    ['compressor_id', 'alert_type', 'severity']
)

compressor_fault_injections = Counter(
    'compressor_fault_injections_total',
    'Total number of fault injections for testing',
    ['compressor_id', 'fault_type']
)

compressor_uptime = Gauge(
    'compressor_uptime_seconds',
    'Uptime of compressor in seconds',
    ['compressor_id', 'compressor_name']
)

compressor_efficiency = Gauge(
    'compressor_efficiency_percentage',
    'Current efficiency percentage of compressor',
    ['compressor_id', 'compressor_name']
)

dashboard_page_views = Counter(
    'dashboard_page_views_total',
    'Total number of dashboard page views',
    ['page_type']
)

prediction_model_accuracy = Gauge(
    'prediction_model_accuracy',
    'Current accuracy of the prediction model',
    ['model_type']
)

maintenance_queue_size = Gauge(
    'maintenance_queue_size',
    'Number of compressors in maintenance queue',
    ['priority_level']
)

sensor_data_processing_time = Histogram(
    'sensor_data_processing_seconds',
    'Time spent processing sensor data',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Add custom metrics endpoint
@server.route('/metrics')
def metrics_endpoint():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Update layout to include metrics information
app.layout = html.Div([
    html.Div(id='login-status', style={'position': 'absolute', 'top': '10px', 'right': '10px'}),
    
    dbc.Row([
        dbc.Col(html.Img(src='/assets/cloud_native.png', height='80px'), width=2),
        dbc.Col(html.H1("Screw Compressor Predictive Maintenance Dashboard", className='text-center'), width=8),
        dbc.Col([
            html.Div([
                html.P("Prometheus Metrics", className='text-center', style={'margin': '0'}),
                html.A("View /metrics", href='/metrics', target='_blank', className='btn btn-info btn-sm')
            ])
        ], width=2)
    ], className='mb-4'),
    
    html.Div(id='main-content', children=[
        # Metrics Overview Section
        dbc.Row([
            dbc.Col([
                html.H3("Metrics Overview"),
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Prometheus Metrics Enabled", className="card-title"),
                            html.P("Real-time metrics are being collected and exposed at /metrics endpoint", className="card-text"),
                            html.P("Metrics include: Sensor readings, Failure probabilities, Alerts, Performance metrics", className="card-text small")
                        ])
                    ])
                ], className='mb-3')
            ], width=12)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                html.H3("System Architecture"),
                html.Img(src='/assets/system_architecture.png', style={'width': '50%'})
            ], width=12, className='text-center')
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                html.H3("Inject Fault for Testing"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Compressor:"),
                        dcc.Dropdown(
                            id='compressor-fault-dropdown',
                            options=[{'label': f'Compressor {i+1} ({f"COMP-{str(i+1).zfill(3)}"})', 'value': i} for i in range(10)],
                            value=None,
                            placeholder="Select a compressor"
                        )
                    ], width=4),
                    dbc.Col([
                        html.Label("Select Fault Type:"),
                        dcc.Dropdown(
                            id='fault-type-dropdown',
                            options=[
                                {'label': 'High Vibration', 'value': 'High Vibration'},
                                {'label': 'Overheating', 'value': 'Overheating'},
                                {'label': 'Low Efficiency', 'value': 'Low Efficiency'},
                                {'label': 'High Current', 'value': 'High Current'}
                            ],
                            value=None,
                            placeholder="Select a fault type"
                        )
                    ], width=4),
                    dbc.Col([
                        html.Br(),
                        html.Button('Inject Fault', id='inject-fault-button', n_clicks=0, className='btn btn-warning'),
                        html.Button('Reset Faults', id='reset-faults-button', n_clicks=0, className='btn btn-secondary', style={'marginLeft': '10px'})
                    ], width=4)
                ]),
                html.Div(id='fault-injection-status', style={'marginTop': '10px'})
            ], width=12)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                html.Label("Select Compressor:"),
                dcc.Dropdown(
                    id='compressor-dropdown',
                    options=[{'label': f'Compressor {i+1} ({f"COMP-{str(i+1).zfill(3)}"})', 'value': i} for i in range(10)],
                    value=0
                )
            ], width=4)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                html.H3("Compressor Analytics"),
                html.P(id='total-compressors', children="Total Compressors Running: 10"),
                html.P(id='maintenance-needed', children="Compressors Needing Maintenance: 0"),
                html.P(id='good-condition', children="Compressors in Good Condition: 10")
            ], width=4),
            dbc.Col(dcc.Graph(id='failure-distribution'), width=8)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                html.H3("Service Engineer Alerts"),
                html.Div(id='warning-alerts', children=[])
            ], width=12)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                html.H3("Predicted Faulty Compressors"),
                dash_table.DataTable(
                    id='faulty-compressors-table',
                    columns=[
                        {'name': 'Unique ID', 'id': 'Unique ID'},
                        {'name': 'Failure Probability', 'id': 'Failure Probability'},
                        {'name': 'Tentative Reason', 'id': 'Tentative Reason'},
                        {'name': 'Suggested Checks', 'id': 'Suggested Checks'}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'whiteSpace': 'pre-line'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{Failure Probability} > 0.7'},
                            'backgroundColor': '#ffcccc',
                            'color': 'black'
                        }
                    ]
                )
            ], width=12)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                html.H3("Maintenance Priority Queue"),
                dash_table.DataTable(
                    id='priority-queue-table',
                    columns=[
                        {'name': 'Unique ID', 'id': 'Unique ID'},
                        {'name': 'Failure Probability', 'id': 'Failure Probability'},
                        {'name': 'Priority Level', 'id': 'Priority Level'},
                        {'name': 'Priority Reason', 'id': 'Priority Reason'},
                        {'name': 'Estimated Downtime', 'id': 'Estimated Downtime'},
                        {'name': 'Action Plan', 'id': 'Action Plan'},
                        {'name': 'RL Decision', 'id': 'RL Decision'}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'whiteSpace': 'pre-line'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{Priority Level} = "Critical"'},
                            'backgroundColor': '#ff3333',
                            'color': 'white'
                        },
                        {
                            'if': {'filter_query': '{Priority Level} = "High"'},
                            'backgroundColor': '#ff9999',
                            'color': 'black'
                        },
                        {
                            'if': {'filter_query': '{Priority Level} = "Medium"'},
                            'backgroundColor': '#ffcccc',
                            'color': 'black'
                        }
                    ]
                )
            ], width=12)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                html.Label("Select Sensor for Trend:"),
                dcc.Dropdown(
                    id='sensor-dropdown',
                    options=[
                        {'label': 'Pressure', 'value': 'Pressure (bar)'},
                        {'label': 'Temperature', 'value': 'Temperature (°C)'},
                        {'label': 'Vibration', 'value': 'Vibration (mm/s)'},
                        {'label': 'Current', 'value': 'Current (A)'},
                        {'label': 'Efficiency', 'value': 'Efficiency (%)'},
                        {'label': 'Vibration Amplitude', 'value': 'Vibration Amplitude (mm)'}
                    ],
                    value='Pressure (bar)'
                )
            ], width=4)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col(dcc.Graph(id='trend-plot'), width=6),
            dbc.Col(dcc.Graph(id='scatter-plot'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='3d-plot'), width=6),
            dbc.Col(dcc.Graph(id='fault-plot'), width=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H3("Recent Sensor Data"),
                dash_table.DataTable(
                    id='data-table',
                    columns=[
                        {'name': 'Time', 'id': 'Time'},
                        {'name': 'Unique ID', 'id': 'Unique ID'},
                        {'name': 'Pressure (bar)', 'id': 'Pressure (bar)'},
                        {'name': 'Temperature (°C)', 'id': 'Temperature (°C)'},
                        {'name': 'Vibration (mm/s)', 'id': 'Vibration (mm/s)'},
                        {'name': 'Current (A)', 'id': 'Current (A)'},
                        {'name': 'Efficiency (%)', 'id': 'Efficiency (%)'},
                        {'name': 'Vibration Amplitude (mm)', 'id': 'Vibration Amplitude (mm)'},
                        {'name': 'Fault Probability', 'id': 'Fault Probability'}
                    ],
                    style_table={'overflowX': 'auto'},
                    page_size=10
                )
            ], width=12)
        ], className='mb-4'),
        
        dcc.Interval(id='interval-component', interval=10000, n_intervals=0)  # 10 seconds interval
    ], style={'display': 'none'}),
    
    html.Footer([
        html.P("Built by Khushal Bendale (GET, ACD D&D) & Dipashri Wagh (Senior Engineer, ACD D&D)", className='text-center'),
        html.P("Prometheus metrics available at /metrics", className='text-center small text-muted')
    ], className='mt-4')
], style={'padding': '20px', 'backgroundColor': '#f8f9fa'})

# Login management setup
login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id):
        self.id = id

users = {'admin': {'password': 'admin123', 'id': '1'}}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Set up logging
logging.basicConfig(
    level=logging.INFO, handlers=[
        logging.FileHandler("logs/dash_app.log"),
        logging.StreamHandler()  # Optional: also print to console
    ])

@server.before_request
def log_request_info():
    logging.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
    # Count page views
    if request.path == '/':
        dashboard_page_views.labels(page_type='dashboard').inc()
    elif request.path == '/login':
        dashboard_page_views.labels(page_type='login').inc()
    elif request.path == '/metrics':
        dashboard_page_views.labels(page_type='metrics').inc()

@server.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            user = User(users[username]['id'])
            login_user(user)
            return redirect('/dashboard')
        else:
            flash('Invalid credentials')
    return '''
        <form method="post">
            <p><input type=text name=username placeholder="Username">
            <p><input type=password name=password placeholder="Password">
            <p><input type=submit value=Login>
        </form>
    '''

@server.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

@server.route('/dashboard')
@login_required
def dashboard():
    return redirect('/')

# Global variables
fault_injection_state = {}
live_data_buffers = {i: pd.DataFrame() for i in range(10)}
compressor_start_times = {i: time.time() for i in range(10)}  # Track uptime

def update_prometheus_metrics(all_data, failures, priority_queue):
    """Update all Prometheus metrics with current data"""
    
    # Update sensor readings for each compressor
    for comp_id in range(10):
        comp_data = all_data[all_data['Compressor ID'] == comp_id]
        if not comp_data.empty:
            latest_data = comp_data.iloc[-1]
            comp_name = f"COMP-{str(comp_id+1).zfill(3)}"
            
            # Update sensor readings
            sensor_metrics = {
                'pressure': ('Pressure (bar)', 'bar'),
                'temperature': ('Temperature (°C)', 'celsius'),
                'vibration': ('Vibration (mm/s)', 'mm_per_sec'),
                'current': ('Current (A)', 'amperes'),
                'vibration_amplitude': ('Vibration Amplitude (mm)', 'mm')
            }
            
            for metric_name, (column_name, unit) in sensor_metrics.items():
                if column_name in latest_data:
                    compressor_sensor_reading.labels(
                        compressor_id=comp_id,
                        compressor_name=comp_name,
                        sensor_type=metric_name,
                        unit=unit
                    ).set(float(latest_data[column_name]))
            
            # Update efficiency
            if 'Efficiency (%)' in latest_data:
                compressor_efficiency.labels(
                    compressor_id=comp_id,
                    compressor_name=comp_name
                ).set(float(latest_data['Efficiency (%)']))
            
            # Update uptime
            uptime_seconds = time.time() - compressor_start_times[comp_id]
            compressor_uptime.labels(
                compressor_id=comp_id,
                compressor_name=comp_name
            ).set(uptime_seconds)
    
    # Update failure probabilities
    for _, failure in failures.iterrows():
        comp_id = int(failure['Unique ID'].split('-')[1]) - 1
        comp_name = failure['Unique ID']
        
        compressor_failure_probability.labels(
            compressor_id=comp_id,
            compressor_name=comp_name
        ).set(float(failure['Failure Probability']))
    
    # Update maintenance queue metrics
    maintenance_queue_size.labels(priority_level='critical').set(
        len(priority_queue[priority_queue['Priority Level'] == 'Critical'])
    )
    maintenance_queue_size.labels(priority_level='high').set(
        len(priority_queue[priority_queue['Priority Level'] == 'High'])
    )
    maintenance_queue_size.labels(priority_level='medium').set(
        len(priority_queue[priority_queue['Priority Level'] == 'Medium'])
    )
    
    # Update model accuracy (placeholder - you can implement actual accuracy calculation)
    prediction_model_accuracy.labels(model_type='failure_prediction').set(0.95)

@app.callback(
    Output('login-status', 'children'),
    Output('main-content', 'style'),
    Input('interval-component', 'n_intervals')
)
def update_login_status(n):
    if current_user.is_authenticated:
        return (
            html.A("Logout", href='/logout', className='btn btn-danger'),
            {'display': 'block'}
        )
    else:
        return (
            html.A("Login", href='/login', className='btn btn-primary'),
            {'display': 'none'}
        )

@app.callback(
    [Output('fault-injection-status', 'children'),
     Output('compressor-dropdown', 'value')],
    [Input('inject-fault-button', 'n_clicks'),
     Input('reset-faults-button', 'n_clicks')],
    [State('compressor-fault-dropdown', 'value'),
     State('fault-type-dropdown', 'value')]
)
def update_fault_injection(inject_clicks, reset_clicks, compressor_id, fault_type):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "No faults injected yet.", 0
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'inject-fault-button':
        if compressor_id is None or fault_type is None:
            return "Please select a compressor and fault type.", dash.no_update
        
        fault_injection_state[compressor_id] = fault_type
        
        # Update Prometheus metrics for fault injection
        compressor_fault_injections.labels(
            compressor_id=compressor_id,
            fault_type=fault_type.lower().replace(' ', '_')
        ).inc()
        
        return f"Injected {fault_type} into Compressor {compressor_id + 1}.", compressor_id
    
    elif button_id == 'reset-faults-button':
        fault_injection_state.clear()
        reset_fault_state()
        global live_data_buffers
        live_data_buffers = {i: pd.DataFrame() for i in range(10)}
        return "All faults reset.", 0

@app.callback(
    [Output('trend-plot', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('3d-plot', 'figure'),
     Output('fault-plot', 'figure'),
     Output('data-table', 'data'),
     Output('failure-distribution', 'figure'),
     Output('total-compressors', 'children'),
     Output('maintenance-needed', 'children'),
     Output('good-condition', 'children'),
     Output('faulty-compressors-table', 'data'),
     Output('priority-queue-table', 'data'),
     Output('warning-alerts', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('compressor-dropdown', 'value'),
     Input('sensor-dropdown', 'value'),
     Input('inject-fault-button', 'n_clicks'),
     Input('reset-faults-button', 'n_clicks')]
)
def update_plots(n, compressor_id, selected_sensor, inject_clicks, reset_clicks):
    global live_data_buffers
    
    # Measure processing time
    start_time = time.time()
    
    new_data = generate_sensor_data(num_compressors=10, n_points=1, fault_injection=fault_injection_state)
    
    for comp_id in range(10):
        comp_data = new_data[new_data['Compressor ID'] == comp_id]
        live_data_buffers[comp_id] = pd.concat([live_data_buffers[comp_id], comp_data], ignore_index=True)
        live_data_buffers[comp_id] = live_data_buffers[comp_id].tail(20)
    
    all_data = pd.concat([live_data_buffers[i] for i in range(10)], ignore_index=True)
    all_data_updated, failures = predict_failures(all_data)

    for comp_id in range(10):
        buffer = live_data_buffers[comp_id]
        updated_data = all_data_updated[all_data_updated['Compressor ID'] == comp_id]
        if not updated_data.empty:
            buffer_indices = buffer.index[-len(updated_data):] if len(updated_data) <= len(buffer) else buffer.index
            updated_indices = updated_data.index[-len(buffer_indices):]
            buffer.loc[buffer_indices, 'Fault Probability'] = updated_data.loc[updated_indices, 'Fault Probability'].values
            buffer.loc[buffer_indices, 'Fault Reason'] = updated_data.loc[updated_indices, 'Fault Reason'].values
            live_data_buffers[comp_id] = buffer

    df = live_data_buffers[compressor_id].copy()

    total_compressors = 10
    maintenance_needed = len(failures[failures['Failure Probability'] > 0.7])
    good_condition = total_compressors - maintenance_needed

    # Create priority queue
    priority_queue = failures[failures['Failure Probability'] > 0.7].copy()
    priority_queue['Priority Level'] = priority_queue['Failure Probability'].apply(
        lambda x: 'Critical' if x > 0.9 else ('High' if x > 0.8 else 'Medium')
    )
    priority_queue['Priority Reason'] = priority_queue.apply(
        lambda row: f"• High failure probability ({row['Failure Probability']:.2f})\n• Reason: {row['Tentative Reason']}", axis=1
    )
    priority_queue['Estimated Downtime'] = priority_queue['Priority Level'].map({
        'Critical': '4 hours',
        'High': '2 hours',
        'Medium': '1 hour'
    })
    priority_queue['Action Plan'] = priority_queue.apply(
        lambda row: f"• Schedule maintenance within: {'1 hour' if row['Priority Level'] == 'Critical' else '4 hours' if row['Priority Level'] == 'High' else '12 hours'}\n"
                    f"• Allocate resources: {'Senior technician + spare parts' if row['Priority Level'] == 'Critical' else 'Technician + tools' if row['Priority Level'] == 'High' else 'Technician'}\n"
                    f"• Notify stakeholders: {'Plant manager, maintenance team' if row['Priority Level'] in ['Critical', 'High'] else 'Maintenance team'}",
        axis=1
    )
    priority_queue['RL Decision'] = priority_queue['Action']
    priority_queue = priority_queue.sort_values(by='Failure Probability', ascending=False)
    
    # Update Prometheus metrics
    update_prometheus_metrics(all_data_updated, failures, priority_queue)
    
    # Count maintenance alerts
    for _, row in priority_queue.iterrows():
        comp_id = int(row['Unique ID'].split('-')[1]) - 1
        severity = row['Priority Level'].lower()
        compressor_maintenance_alerts.labels(
            compressor_id=comp_id,
            alert_type='maintenance_required',
            severity=severity
        ).inc()
    
    # Record processing time
    processing_time = time.time() - start_time
    sensor_data_processing_time.observe(processing_time)

    # Create plots
    df_plot = df.tail(20)
    trend_fig = px.line(df_plot, x='Time', y=selected_sensor, title=f'{selected_sensor} Trend - Compressor {compressor_id + 1}')
    trend_fig.update_layout(template='plotly_white')

    scatter_fig = px.scatter(df_plot, x='Vibration (mm/s)', y='Current (A)', color='Pressure (bar)',
                            title=f'Vibration vs Current - Compressor {compressor_id + 1}', size='Temperature (°C)')
    scatter_fig.update_layout(template='plotly_white')

    plot_3d_fig = px.scatter_3d(df_plot, x='Pressure (bar)', y='Temperature (°C)', z='Vibration (mm/s)',
                                color='Current (A)', title=f'3D Sensor Analysis - Compressor {compressor_id + 1}')
    plot_3d_fig.update_layout(template='plotly_white')

    fault_fig = px.line(df_plot, x='Time', y='Fault Probability', title=f'Fault Probability - Compressor {compressor_id + 1}')
    fault_fig.update_layout(template='plotly_white', yaxis_range=[0, 1])

    failure_dist_fig = px.bar(
        failures,
        x='Unique ID',
        y='Failure Probability',
        title='Failure Probability Per Compressor',
        color='Failure Probability',
        color_continuous_scale='Reds',
        range_y=[0, 1]
    )
    failure_dist_fig.update_layout(
        template='plotly_white',
        xaxis_title="Compressor Unique ID",
        yaxis_title="Failure Probability",
        showlegend=False
    )
    failure_dist_fig.add_shape(
        type='line',
        x0=-0.5,
        x1=len(failures)-0.5,
        y0=0.7,
        y1=0.7,
        line=dict(color='black', dash='dash')
    )
    failure_dist_fig.add_annotation(
        x=len(failures)/2,
        y=0.72,
        text="Threshold (0.7)",
        showarrow=False,
        font=dict(color='black')
    )

    table_data = df.tail(10).to_dict('records')
    faulty_compressors_data = failures[['Unique ID', 'Failure Probability', 'Tentative Reason', 'Suggested Checks']].to_dict('records')
    priority_queue_data = priority_queue[['Unique ID', 'Failure Probability', 'Priority Level', 'Priority Reason', 'Estimated Downtime', 'Action Plan', 'RL Decision']].to_dict('records')

    # Generate alerts
    alerts = []
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for _, row in priority_queue.iterrows():
        alert = dbc.Alert([
            html.H4(f"Warning: Compressor {row['Unique ID']} Requires Immediate Attention!", className="alert-heading"),
            html.P([
                f"Detected at: {current_time}",
                html.Br(),
                f"Fault Probability: {row['Failure Probability']:.2f}",
                html.Br(),
                f"Tentative Reason: {row['Tentative Reason']}",
                html.Br(),
                f"Action Recommended: {row['Action']}"
            ])
        ], color="danger", dismissable=False, className="mb-2")
        alerts.append(alert)
    
    if not alerts:
        alerts = [html.P("No critical faults detected at this time.", style={'color': 'green'})]

    return (trend_fig, scatter_fig, plot_3d_fig, fault_fig, table_data,
            failure_dist_fig, f"Total Compressors Running: {total_compressors}",
            f"Compressors Needing Maintenance: {maintenance_needed}",
            f"Compressors in Good Condition: {good_condition}",
            faulty_compressors_data, priority_queue_data, alerts)

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")