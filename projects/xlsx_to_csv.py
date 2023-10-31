import pandas as pd


input_file = 'mbpp.xlsx'
output_file = 'mbpp.csv'

# Read the XLSX file into a DataFrame.
df = pd.read_excel(input_file)

# Convert the DataFrame to a CSV file.
df.to_csv(output_file, index=False)

print(f'Conversion from {input_file} to {output_file} complete.')

