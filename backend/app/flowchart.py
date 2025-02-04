from graphviz import Digraph

# Create a directed graph
dot = Digraph(comment='Flowchart of JobStreet Job Search System Using K-Means Clustering')

# Add nodes (steps of your flowchart) with specific shapes
dot.node('A', 'Start', shape='ellipse')  # Start node as ellipse
dot.node('B', 'Load Data (Job Listings)', shape='box')  # Step as box
dot.node('C', 'Preprocess Data (Normalize and Clean)', shape='box')  # Step as box
dot.node('D', 'Apply K-Means Clustering', shape='box')  # Step as box
dot.node('E', 'Assign Job Categories to Users', shape='box')  # Step as box
dot.node('F', 'Generate Job Recommendations', shape='box')  # Step as box
dot.node('G', 'Evaluate Recommendation Accuracy', shape='diamond')  # Decision step as diamond
dot.node('H', 'End', shape='ellipse')  # End node as ellipse

# Add edges (connections between steps)
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G')
dot.edge('G', 'H')

# Render and display the flowchart
dot.render('jobstreet_flowchart_complete', format='png', view=True)
