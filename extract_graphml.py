import datetime
import networkx as nx
import os
import pandas as pd
import argparse
import numpy as np

## Given an output directory, read the entites.parquet file and a relationships.parquet file.

def read_parquet_files(output_dir):
    """
    Read the entities and relationships parquet files from the output directory.
    """
    entities_path = os.path.join(output_dir, "entities.parquet")
    relationships_path = os.path.join(output_dir, "relationships.parquet")

    # Read the parquet files
    entities_df = pd.read_parquet(entities_path)
    relationships_df = pd.read_parquet(relationships_path)
    
    for col in entities_df.columns:
        if entities_df[col].dtype == np.dtype('O'):
            entities_df[col] = entities_df[col].apply(lambda x: ";".join(x.tolist()) if isinstance(x, np.ndarray) else x)
    
    for col in relationships_df.columns:
        if relationships_df[col].dtype == np.dtype('O'):
            relationships_df[col] = relationships_df[col].apply(lambda x: ";".join(x.tolist()) if isinstance(x, np.ndarray) else x)

    return entities_df, relationships_df

def fix_time_weight(target_str):
    
    # target_str is a date/time string in the format MAR19 23:00.
    # calculate how many minutes from CURRENT TIME it is to target_str
    try:
        # Parse the target string into a datetime object
        target_time = datetime.datetime.strptime(target_str, "[%b%d %H:%M]")
        
        # Get the current time
        current_time = datetime.datetime.now()
        
        # Adjust the year of the target time to match the current year
        target_time = target_time.replace(year=current_time.year)
        
        # Calculate the difference in minutes
        time_difference = (target_time - current_time).total_seconds() / 60 / 10000
        
        # Return the absolute value of the time difference
        return 15.0 - abs(time_difference)
    except ValueError as e:
        print(f"Error parsing time: {e}")
        return float('inf')  # Return a large value if parsing fails
    
    
    

def create_graph(entities_df, relationships_df) -> nx.Graph:
    """
    Create a graph from the entities and relationships dataframes.
    """
    # Create an undirected graph respecting the weights of the relationships
    G = nx.Graph()
    
    print ("Adding nodes:")
    
    # Add nodes with their attributes
    for _, row in entities_df.iterrows():
        
        if row['type'] != "" and row['type'] is not None:
            print (f"\t{row['title']}")
            
            # set row['x_extract'] equal to row['x'] and remove row['x'] from the row
            row['x_extract'] = row['x']
            row['y_extract'] = row['y']
                       
            G.add_node(row['title'], **row.to_dict())
           
    print ("\n\nAdding edges:")
    
    # Add edges with their attributes
    for _, row in relationships_df.iterrows():
        
        # split row['description'] by the delimiter '|' and store the first piece in a variable named type and the second in a variable named description
        type, description = row['description'].split('|', 1) if '|' in row['description'] else ('UNKNOWN', row['description'])

        row['type']=type
        row['description']=description
        
        #if type == "observed_time":
            #row['weight'] = fix_time_weight(row['target'])
        #    print (f"\n\n{row['target']}: {fix_time_weight(row['target'])}")
            
        
        if type != "UNKNOWN":
            #print (f"\t{row['source']} --> {row['target']}: {row.to_dict()}")
            print (f"\t{row['source']} --> {row['target']}: {row['type']}")
            
            row['label'] = f"{row['source']} TO {row['target']}"
            
            if type == "observed_time":
                #row['weight'] = fix_time_weight(row['target'])
                time = row['target'] if "MAR" in row['target'] else row['source']
                
                row['weight'] = ( 5 * fix_time_weight(time) ) - 40.0                
                row['weight'] /= 10.0
                
            
            G.add_edge(row['source'], row['target'], **row.to_dict()) 
        
    return G


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate a GraphML file from parquet files.")
    parser.add_argument("--dir", "-d", required=True, help="Output directory containing parquet files.")
    parser.add_argument("--output", "-o", default="graph.graphml", help="Filename for the exported GraphML file. Default is 'graph.graphml'.")
    args = parser.parse_args()

    # Read the parquet files
    entities_df, relationships_df = read_parquet_files(args.dir)

    # Create the graph
    G = create_graph(entities_df, relationships_df)

    # Export the graph to a GraphML file
    output_path = os.path.join(args.dir, args.output)
    nx.write_graphml(G, output_path)
    print(f"GraphML file has been written to {output_path}")


main()
