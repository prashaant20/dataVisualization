# Step 1: Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np   # For numerical operations
import plotly.express as px  # For interactive visualizations
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.linear_model import LinearRegression  # For creating a linear regression model
from sklearn.metrics import mean_squared_error  # For evaluating model performance
from dash import Dash, dcc, html  # For creating the dashboard
from dash.dependencies import Input, Output  # For handling user input and updating dashboard components

# Step 2: Generate sample sales data (replace with your own dataset in real projects)
data = {
    'Date': pd.date_range(start='2023-01-01', periods=100),
    'Product': np.random.choice(['Laptop', 'Smartphone', 'Tablet'], 100),
    'Quantity': np.random.randint(1, 10, 100),
    'Price': np.random.uniform(500, 1500, 100),
    'Sales': np.random.uniform(500, 5000, 100)
}
df = pd.DataFrame(data)  # Create a DataFrame to store the data

# Step 3: Feature engineering - Extract month and year from Date column
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Step 4: Machine learning model setup (linear regression) for predictive analysis
X = df[['Quantity', 'Price']]  # Features for prediction
y = df['Sales']  # Target variable to predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and testing sets
model = LinearRegression()  # Create a linear regression model
model.fit(X_train, y_train)  # Train the model
predictions = model.predict(X_test)  # Make predictions
mse = mean_squared_error(y_test, predictions)  # Calculate Mean Squared Error for model evaluation

# Step 5: Initialize the Dash app
app = Dash(__name__)  # Create an instance of the Dash class

# Step 6: Define the layout of the dashboard
app.layout = html.Div([
    html.H1('Sales Analysis Dashboard', style={'textAlign': 'center'}),  # Dashboard title
    
    # Graph to show sales distribution by product
    dcc.Graph(id='sales-by-product', figure={}),
    
    # Graph to show monthly sales trends
    dcc.Graph(id='monthly-sales', figure={}),
    
    # Scatter plot to show quantity vs sales with year coloring
    dcc.Graph(id='scatter-plot', figure={}),
    
    # Section to display model evaluation metrics
    html.Div([
        html.H2('Predictive Model Evaluation'),  # Section title
        html.P(f'Mean Squared Error: {mse:.2f}')  # Display Mean Squared Error value
    ], style={'textAlign': 'center'}),  # Center align text
])

# Step 7: Callback to update sales-by-product graph based on hover data from sales-by-product graph
@app.callback(
    Output('sales-by-product', 'figure'),  # Output component: sales-by-product graph
    [Input('sales-by-product', 'hoverData')]  # Input component: hover data from sales-by-product graph
)
def update_sales_by_product(hoverData):
    try:
        # Default to first product if hoverData is None, otherwise use selected product
        if hoverData is None:
            selected_product = df['Product'].iloc[0]
        else:
            selected_product = hoverData['points'][0]['label']
        
        # Filter data for selected product
        filtered_data = df[df['Product'] == selected_product]
        
        # Create bar chart for sales distribution by product using Plotly Express
        sales_by_product_fig = px.bar(filtered_data, x='Product', y='Sales', 
                                      title=f'Sales Distribution for {selected_product}')
        
        return sales_by_product_fig  # Return the figure to update sales-by-product graph
    
    except Exception as e:
        print(f"Error updating sales-by-product figure: {str(e)}")  # Print error message if any error occurs
        return {}  # Return empty figure in case of error

# Step 8: Callback to update monthly-sales graph based on hover data from sales-by-product graph
@app.callback(
    Output('monthly-sales', 'figure'),  # Output component: monthly-sales graph
    [Input('sales-by-product', 'hoverData')]  # Input component: hover data from sales-by-product graph
)
def update_monthly_sales(hoverData):
    try:
        # Default to first product if hoverData is None, otherwise use selected product
        if hoverData is None:
            selected_product = df['Product'].iloc[0]
        else:
            selected_product = hoverData['points'][0]['label']
        
        # Filter data for selected product
        filtered_data = df[df['Product'] == selected_product]
        
        # Aggregate monthly sales for selected product
        monthly_sales = filtered_data.groupby('Month')['Sales'].sum().reset_index()
        
        # Create line chart for monthly sales trends using Plotly Express
        monthly_sales_fig = px.line(monthly_sales, x='Month', y='Sales', 
                                    title=f'Monthly Sales for {selected_product}')
        
        return monthly_sales_fig  # Return the figure to update monthly-sales graph
    
    except Exception as e:
        print(f"Error updating monthly-sales figure: {str(e)}")  # Print error message if any error occurs
        return {}  # Return empty figure in case of error

# Step 9: Callback to update scatter-plot based on hover data from sales-by-product graph
@app.callback(
    Output('scatter-plot', 'figure'),  # Output component: scatter-plot graph
    [Input('sales-by-product', 'hoverData')]  # Input component: hover data from sales-by-product graph
)
def update_scatter_plot(hoverData):
    try:
        # Default to first product if hoverData is None, otherwise use selected product
        if hoverData is None:
            selected_product = df['Product'].iloc[0]
        else:
            selected_product = hoverData['points'][0]['label']
        
        # Filter data for selected product
        filtered_data = df[df['Product'] == selected_product]
        
        # Create scatter plot for Quantity vs Sales with year coloring using Plotly Express
        scatter_plot_fig = px.scatter(filtered_data, x='Quantity', y='Sales', color='Year', 
                                      title=f'Scatter Plot: Quantity vs Sales for {selected_product}',
                                      hover_name='Product')
        
        return scatter_plot_fig  # Return the figure to update scatter-plot
    
    except Exception as e:
        print(f"Error updating scatter-plot figure: {str(e)}")  # Print error message if any error occurs
        return {}  # Return empty figure in case of error

# Step 10: Main entry point to run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)  # Run the Dash app in debug mode

