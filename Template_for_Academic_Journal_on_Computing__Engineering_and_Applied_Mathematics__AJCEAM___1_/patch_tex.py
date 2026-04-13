import re

file_path = '/Users/hamza.aalaou/MLBA/Template_for_Academic_Journal_on_Computing__Engineering_and_Applied_Mathematics__AJCEAM___1_/ajceam-paper.tex'
with open(file_path, 'r') as f:
    text = f.read()

# Add float package
if '\\usepackage{float}' not in text:
    text = text.replace('\\newcommand{\\vect}[1]{\\mathbf{#1}}', '\\usepackage{float}\n\\newcommand{\\vect}[1]{\\mathbf{#1}}')

# Change figure* to figure
text = text.replace('\\begin{figure*}', '\\begin{figure}')
text = text.replace('\\end{figure*}', '\\end{figure}')

# Change [!tb] etc. to [H] for strict inline placement
text = re.sub(r'\\begin\{figure\}\s*\[.*?\]', r'\\begin{figure}[H]', text)

# Ensure \textwidth is \columnwidth now that they are not spanning both columns
text = text.replace('\\includegraphics[width=\\textwidth]', '\\includegraphics[width=\\columnwidth]')

# Update Table RMSE values
text = text.replace('Lasso Regression (Tuned) & 0.144 \\\\', 'Lasso Regression (Tuned) & 0.201 \\\\')
text = text.replace('Random Forest & 0.141 \\\\', 'Random Forest & 0.169 \\\\')
text = text.replace('XGBoost (Ensemble) & \\textbf{0.133} \\\\', 'XGBoost (Ensemble) & \\textbf{0.163} \\\\')

text = text.replace('\\begin{table}[!b]', '\\begin{table}[H]')

with open(file_path, 'w') as f:
    f.write(text)

