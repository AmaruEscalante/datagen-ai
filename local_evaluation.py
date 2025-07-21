"""
Local evaluation system for Gretel Data Designer bypass.
Provides statistical analysis and HTML report generation matching Gretel's report format.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from jinja2 import Template

logger = logging.getLogger(__name__)


class LocalEvaluationEngine:
    """Local statistical analysis and evaluation engine."""
    
    def __init__(self):
        self.analysis_results = {}
        self.plotly_charts = []
    
    def analyze_dataset(self, df: pd.DataFrame, aidd_metadata=None) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis of the dataset."""
        logger.info("🔍 Starting local dataset evaluation...")
        
        analysis = {
            'dataset_preview': self._analyze_dataset_preview(df),
            'dataset_schema': self._analyze_dataset_schema(df),
            'dataset_statistics': self._analyze_dataset_statistics(df),
            'column_distributions': self._analyze_column_distributions(df, aidd_metadata),
            'correlations': self._analyze_correlations(df),
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'rows': len(df),
                'columns': len(df.columns),
                'report_id': str(uuid.uuid4())[:8]
            }
        }
        
        # Add AIDD-specific metadata if available
        if aidd_metadata:
            analysis['aidd_metadata'] = {
                'sampler_columns': getattr(aidd_metadata, 'sampler_columns', []),
                'llm_text_columns': getattr(aidd_metadata, 'llm_text_columns', []),
                'llm_judge_columns': getattr(aidd_metadata, 'llm_judge_columns', [])
            }
        
        self.analysis_results = analysis
        logger.info("✅ Dataset analysis completed")
        return analysis
    
    def _analyze_dataset_preview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate dataset preview (first 5 rows)."""
        preview_df = df.head(5).copy()
        
        # Convert to dict with proper handling of long text
        preview_data = []
        for idx, row in preview_df.iterrows():
            row_data = {}
            for col, val in row.items():
                # Truncate very long values for display
                str_val = str(val) if val is not None else "null"
                if len(str_val) > 200:
                    str_val = str_val[:197] + "..."
                row_data[col] = str_val
            preview_data.append(row_data)
        
        return {
            'sample_records': preview_data,
            'total_records': len(df),
            'columns': list(df.columns)
        }
    
    def _analyze_dataset_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset schema and column properties."""
        schema_info = []
        
        for col in df.columns:
            col_series = df[col]
            
            # Calculate basic statistics
            null_count = col_series.isnull().sum()
            null_percentage = (null_count / len(df)) * 100
            
            # Estimate token count (simple word count * 1.3 as approximation)
            non_null_values = col_series.dropna().astype(str)
            if len(non_null_values) > 0:
                avg_length = non_null_values.str.len().mean()
                avg_words = non_null_values.str.split().str.len().mean()
                estimated_tokens = avg_words * 1.3 if not pd.isna(avg_words) else 0
            else:
                avg_length = 0
                avg_words = 0
                estimated_tokens = 0
            
            # Determine data type
            dtype = str(col_series.dtype)
            if dtype == 'object':
                # Try to infer more specific type
                sample_values = col_series.dropna().head(10)
                if len(sample_values) > 0:
                    if sample_values.str.isdigit().all():
                        inferred_type = "numeric (as text)"
                    elif sample_values.str.len().var() < 5:  # Low variance in length
                        inferred_type = "categorical"
                    else:
                        inferred_type = "text"
                else:
                    inferred_type = "text"
            else:
                inferred_type = dtype
            
            schema_info.append({
                'column': col,
                'type': inferred_type,
                'null_percentage': round(null_percentage, 2),
                'avg_length': round(avg_length, 1) if not pd.isna(avg_length) else 0,
                'avg_tokens': round(estimated_tokens, 1),
                'avg_words': round(avg_words, 1) if not pd.isna(avg_words) else 0,
                'unique_values': col_series.nunique(),
                'sample_values': col_series.dropna().head(3).tolist()
            })
        
        return {
            'columns': schema_info,
            'total_columns': len(df.columns)
        }
    
    def _analyze_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate high-level dataset statistics."""
        # Data completeness
        total_cells = df.size
        non_null_cells = df.count().sum()
        completeness = (non_null_cells / total_cells) * 100
        
        # Estimate total tokens
        total_tokens = 0
        for col in df.columns:
            col_series = df[col].dropna().astype(str)
            if len(col_series) > 0:
                word_counts = col_series.str.split().str.len().sum()
                total_tokens += word_counts * 1.3  # Approximate token count
        
        # Row uniqueness (exact duplicates)
        unique_rows = len(df.drop_duplicates())
        row_uniqueness = (unique_rows / len(df)) * 100
        
        # Semantic uniqueness (approximate - based on string similarity of concatenated rows)
        sample_size = min(100, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        concatenated = sample_df.astype(str).apply(' '.join, axis=1)
        semantic_unique = concatenated.nunique()
        semantic_uniqueness = (semantic_unique / sample_size) * 100
        
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'completeness_percentage': round(completeness, 2),
            'total_tokens': int(total_tokens),
            'unique_rows': unique_rows,
            'row_uniqueness_percentage': round(row_uniqueness, 2),
            'semantic_uniqueness_percentage': round(semantic_uniqueness, 2),
            'avg_tokens_per_row': round(total_tokens / len(df), 1) if len(df) > 0 else 0
        }
    
    def _analyze_column_distributions(self, df: pd.DataFrame, aidd_metadata=None) -> Dict[str, Any]:
        """Analyze distributions of categorical columns (non-LLM columns only)."""
        distributions = {}
        
        # Get LLM columns to exclude from distributions
        llm_columns = set()
        if aidd_metadata:
            llm_columns.update(getattr(aidd_metadata, 'llm_text_columns', []))
            llm_columns.update(getattr(aidd_metadata, 'llm_judge_columns', []))
        
        for col in df.columns:
            # Skip LLM-generated columns
            if col in llm_columns:
                continue
                
            col_series = df[col].dropna()
            
            # Only analyze columns that appear categorical
            unique_vals = col_series.nunique()
            if unique_vals <= 20 and len(col_series) > 0:  # Reasonable for bar chart
                value_counts = col_series.value_counts().head(10)  # Top 10
                
                distributions[col] = {
                    'values': value_counts.index.tolist(),
                    'counts': value_counts.values.tolist(),
                    'percentages': (value_counts / len(col_series) * 100).round(1).tolist(),
                    'total_unique': unique_vals,
                    'chart_id': str(uuid.uuid4()),
                    'chart_type': 'bar'
                }
        
        return distributions
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between categorical columns."""
        categorical_cols = []
        
        # Identify categorical columns
        for col in df.columns:
            if df[col].nunique() <= 20:  # Treat as categorical
                categorical_cols.append(col)
        
        if len(categorical_cols) < 2:
            return {'correlation_matrix': None, 'message': 'Not enough categorical columns for correlation analysis'}
        
        # Create correlation matrix using Cramér's V for categorical data
        correlation_matrix = []
        col_names = categorical_cols
        
        for i, col1 in enumerate(categorical_cols):
            row = []
            for j, col2 in enumerate(categorical_cols):
                if i == j:
                    correlation = 1.0
                else:
                    correlation = self._cramers_v(df[col1], df[col2])
                row.append(correlation)
            correlation_matrix.append(row)
        
        return {
            'correlation_matrix': correlation_matrix,
            'column_names': col_names,
            'chart_id': str(uuid.uuid4()),
            'chart_type': 'heatmap'
        }
    
    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate Cramér's V correlation coefficient for categorical variables."""
        try:
            # Create contingency table
            crosstab = pd.crosstab(x, y)
            
            # Chi-square test
            from scipy.stats import chi2_contingency
            chi2, _, _, _ = chi2_contingency(crosstab)
            
            # Calculate Cramér's V
            n = crosstab.sum().sum()
            min_dim = min(len(crosstab), len(crosstab.columns)) - 1
            
            if min_dim == 0:
                return 0.0
            
            cramers_v = np.sqrt(chi2 / (n * min_dim))
            return min(cramers_v, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0.0
    
    def generate_plotly_charts(self) -> List[Dict[str, Any]]:
        """Generate Plotly chart configurations for visualizations."""
        charts = []
        
        # Generate distribution charts with dual-axis (count + percentage)
        if 'column_distributions' in self.analysis_results:
            for col_name, dist_data in self.analysis_results['column_distributions'].items():
                chart_config = {
                    'id': dist_data['chart_id'],
                    'type': 'bar',
                    'title': f"Distribution - {col_name}",
                    'data': [
                        {
                            'x': dist_data['values'],
                            'y': dist_data['counts'],
                            'type': 'bar',
                            'marker': {'color': '#A051FA'},
                            'name': 'Count',
                            'opacity': 0.75,
                            'yaxis': 'y',
                            'text': [f"{count}" for count in dist_data['counts']],
                            'textposition': 'auto'
                        },
                        {
                            'x': dist_data['values'],
                            'y': dist_data['percentages'],
                            'type': 'scatter',
                            'mode': 'lines+markers',
                            'marker': {'color': '#FF6B6B', 'size': 8},
                            'line': {'color': '#FF6B6B', 'width': 3},
                            'name': 'Percentage',
                            'yaxis': 'y2',
                            'text': [f"{pct}%" for pct in dist_data['percentages']],
                            'textposition': 'top center'
                        }
                    ],
                    'layout': {
                        'title': {'text': f'{col_name} Distribution', 'x': 0.5},
                        'xaxis': {'title': {'text': col_name}},
                        'yaxis': {
                            'title': {'text': 'Count'},
                            'side': 'left',
                            'showgrid': True
                        },
                        'yaxis2': {
                            'title': {'text': 'Percentage (%)'},
                            'side': 'right',
                            'overlaying': 'y',
                            'showgrid': False,
                            'zeroline': False
                        },
                        'showlegend': True,
                        'legend': {'x': 0, 'y': 1},
                        'width': 700,
                        'height': 450,
                        'hovermode': 'x unified'
                    }
                }
                charts.append(chart_config)
        
        # Generate correlation heatmap
        if 'correlations' in self.analysis_results and self.analysis_results['correlations'].get('correlation_matrix'):
            corr_data = self.analysis_results['correlations']
            chart_config = {
                'id': corr_data['chart_id'],
                'type': 'heatmap',
                'title': 'Column Correlations',
                'data': [{
                    'z': corr_data['correlation_matrix'],
                    'x': corr_data['column_names'],
                    'y': corr_data['column_names'],
                    'type': 'heatmap',
                    'colorscale': 'Viridis',
                    'showscale': True,
                    'hoverongaps': False
                }],
                'layout': {
                    'title': {'text': 'Column Correlations', 'x': 0.5},
                    'width': 800,
                    'height': 600,
                    'xaxis': {'title': 'Columns'},
                    'yaxis': {'title': 'Columns'}
                }
            }
            charts.append(chart_config)
        
        self.plotly_charts = charts
        return charts


class LocalEvaluationSettings:
    """Local evaluation configuration settings."""
    
    def __init__(self, settings_dict: Optional[Dict[str, Any]] = None):
        settings = settings_dict or {}
        
        self.llm_judge_columns = settings.get('llm_judge_columns', [])
        self.validation_columns = settings.get('validation_columns', [])
        self.defined_categorical_columns = settings.get('defined_categorical_columns', [])
        self.include_correlations = settings.get('include_correlations', True)
        self.include_distributions = settings.get('include_distributions', True)
        self.max_categorical_values = settings.get('max_categorical_values', 20)


def create_evaluation_report(df: pd.DataFrame, 
                           aidd_metadata=None,
                           settings: Optional[LocalEvaluationSettings] = None,
                           output_path: Optional[str] = None) -> str:
    """Create a complete evaluation report matching Gretel's format."""
    
    # Initialize evaluation engine
    engine = LocalEvaluationEngine()
    
    # Perform analysis
    analysis = engine.analyze_dataset(df, aidd_metadata)
    
    # Generate charts
    charts = engine.generate_plotly_charts()
    
    # Generate HTML report
    report_generator = LocalReportGenerator()
    html_content = report_generator.generate_html_report(analysis, charts)
    
    # Save report if path specified
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"📊 Evaluation report saved to: {output_path}")
    
    return html_content


class LocalReportGenerator:
    """Generates HTML reports matching Gretel's design."""
    
    def generate_html_report(self, analysis: Dict[str, Any], charts: List[Dict[str, Any]]) -> str:
        """Generate complete HTML report."""
        
        # Custom filters for Jinja2
        def format_datetime(timestamp_str):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d at %H:%M:%S')
            except:
                return str(timestamp_str)
        
        def truncate_text(text, length=100):
            if text is None:
                return "null"
            if len(str(text)) > length:
                return str(text)[:length-3] + "..."
            return str(text)
        
        # Basic template - we'll expand this to match Gretel's styling
        template_str = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Local Data Designer Report</title>
    <link href='https://fonts.googleapis.com/css?family=Inter' rel='stylesheet'>
    <script src="https://cdn.plot.ly/plotly-2.35.3.min.js"></script>
    <style>
        :root {
            --primary: #25212B;
            --secondary: #646071;
            --accent: #5351DE;
            --success: #E0F9F1;
            --success-dark: #1D9F84;
        }
        
        body {
            font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 20px;
            color: var(--primary);
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .header {
            padding: 24px;
            border-bottom: 1px solid #e9ecef;
            background: linear-gradient(135deg, var(--accent) 0%, #7c3aed 100%);
            color: white;
            border-radius: 8px 8px 0 0;
        }
        
        .header h1 {
            margin: 0;
            font-size: 24px;
            font-weight: 700;
        }
        
        .content {
            padding: 24px;
        }
        
        .section {
            margin-bottom: 32px;
            padding-bottom: 24px;
            border-bottom: 1px solid #e9ecef;
        }
        
        .section:last-child {
            border-bottom: none;
        }
        
        .section-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--primary);
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 16px;
            border-radius: 8px;
            border-left: 4px solid var(--accent);
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--accent);
        }
        
        .metric-label {
            font-size: 14px;
            color: var(--secondary);
            margin-top: 4px;
        }
        
        .chart-container {
            margin: 20px 0;
            padding: 16px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }
        
        .table th,
        .table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }
        
        .table th {
            background: #f8f9fa;
            font-weight: 600;
            color: var(--primary);
        }
        
        .preview-table {
            overflow-x: auto;
            margin-top: 16px;
        }
        
        .timestamp {
            text-align: center;
            color: var(--secondary);
            font-size: 12px;
            margin-top: 24px;
            padding-top: 16px;
            border-top: 1px solid #e9ecef;
        }

        .dropdown-container {
            margin-bottom: 24px;
        }

        .dropdown-select {
            padding: 8px 12px;
            border: 2px solid var(--accent);
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            background: white;
            color: var(--primary);
            cursor: pointer;
            min-width: 200px;
        }

        .dropdown-select:focus {
            outline: none;
            border-color: #7c3aed;
            box-shadow: 0 0 0 3px rgba(92, 81, 222, 0.1);
        }

        .distribution-content {
            display: none;
            margin-top: 20px;
        }

        .distribution-content.active {
            display: block;
        }

        .distribution-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 24px;
            margin-top: 16px;
        }

        .distribution-table-container {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
        }

        .distribution-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }

        .distribution-table th {
            background: var(--accent);
            color: white;
            padding: 10px;
            text-align: left;
            font-weight: 600;
        }

        .distribution-table td {
            padding: 8px 10px;
            border-bottom: 1px solid #e9ecef;
        }

        .distribution-table tr:nth-child(even) {
            background: white;
        }

        .distribution-chart-container {
            background: white;
            border-radius: 8px;
            padding: 16px;
            border: 1px solid #e9ecef;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Local Data Designer Report</h1>
            <p>Generated on {{ format_datetime(analysis.metadata.analysis_timestamp) }}</p>
        </div>
        
        <div class="content">
            <!-- Dataset Statistics -->
            <div class="section">
                <h2 class="section-title">Dataset Statistics</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{{ analysis.dataset_statistics.rows }}</div>
                        <div class="metric-label">Total Rows</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ analysis.dataset_statistics.columns }}</div>
                        <div class="metric-label">Total Columns</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ analysis.dataset_statistics.completeness_percentage }}%</div>
                        <div class="metric-label">Data Completeness</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ analysis.dataset_statistics.row_uniqueness_percentage }}%</div>
                        <div class="metric-label">Row Uniqueness</div>
                    </div>
                </div>
            </div>
            
            <!-- Dataset Schema -->
            <div class="section">
                <h2 class="section-title">Dataset Schema</h2>
                <div class="preview-table">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Column</th>
                                <th>Type</th>
                                <th>Null %</th>
                                <th>Avg Length</th>
                                <th>Avg Tokens</th>
                                <th>Unique Values</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for col in analysis.dataset_schema.columns %}
                            <tr>
                                <td><strong>{{ col.column }}</strong></td>
                                <td>{{ col.type }}</td>
                                <td>{{ col.null_percentage }}%</td>
                                <td>{{ col.avg_length }}</td>
                                <td>{{ col.avg_tokens }}</td>
                                <td>{{ col.unique_values }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Dataset Preview -->
            <div class="section">
                <h2 class="section-title">Dataset Preview</h2>
                <p>Sample of {{ analysis.dataset_preview.sample_records | length }} records from the dataset:</p>
                <div class="preview-table">
                    <table class="table">
                        <thead>
                            <tr>
                                {% for col in analysis.dataset_preview.columns %}
                                <th>{{ col }}</th>
                                {% endfor %}
                            </tr>
                        </thead>
                        <tbody>
                            {% for record in analysis.dataset_preview.sample_records %}
                            <tr>
                                {% for col in analysis.dataset_preview.columns %}
                                <td>{{ truncate_text(record[col], 100) }}</td>
                                {% endfor %}
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Column Distributions -->
            {% if analysis.column_distributions %}
            <div class="section">
                <h2 class="section-title">Column Distributions</h2>
                <p>Select a column to view its distribution analysis:</p>
                
                <div class="dropdown-container">
                    <select class="dropdown-select" id="columnSelector" onchange="showDistribution(this.value)">
                        <option value="">Select a column...</option>
                        {% for col_name in analysis.column_distributions.keys() %}
                        <option value="{{ col_name }}">{{ col_name }}</option>
                        {% endfor %}
                    </select>
                </div>

                {% for col_name, dist in analysis.column_distributions.items() %}
                <div class="distribution-content" id="dist-{{ col_name }}">
                    <h3>{{ col_name }} Distribution</h3>
                    
                    <div class="distribution-grid">
                        <!-- Data Table -->
                        <div class="distribution-table-container">
                            <h4 style="margin-top: 0; color: var(--primary);">Distribution Table</h4>
                            <table class="distribution-table">
                                <thead>
                                    <tr>
                                        <th>Category</th>
                                        <th>Count</th>
                                        <th>Percentage</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for value in dist['values'] %}
                                    {% set outer_loop = loop %}
                                    <tr>
                                        <td><strong>{{ value }}</strong></td>
                                        <td>{{ dist['counts'][outer_loop.index0] }}</td>
                                        <td>{{ dist['percentages'][outer_loop.index0] }}%</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>

                        <!-- Chart -->
                        <div class="distribution-chart-container">
                            <h4 style="margin-top: 0; color: var(--primary);">Distribution Chart</h4>
                            <div id="{{ dist['chart_id'] }}" style="height:400px;"></div>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            <!-- Correlations -->
            {% if analysis.correlations.correlation_matrix %}
            <div class="section">
                <h2 class="section-title">Column Correlations</h2>
                <div class="chart-container">
                    <div id="{{ analysis.correlations['chart_id'] }}" style="height:600px;"></div>
                </div>
            </div>
            {% endif %}
            
            <div class="timestamp">
                Report generated by Local Data Designer Evaluation Engine<br>
                Analysis ID: {{ analysis.metadata.report_id }}
            </div>
        </div>
    </div>
    
    <script>
        // Function to show/hide distribution content based on dropdown selection
        function showDistribution(columnName) {
            // Hide all distribution content
            const allDistributions = document.querySelectorAll('.distribution-content');
            allDistributions.forEach(dist => {
                dist.classList.remove('active');
            });
            
            // Show selected distribution
            if (columnName) {
                const selectedDist = document.getElementById('dist-' + columnName);
                if (selectedDist) {
                    selectedDist.classList.add('active');
                }
            }
        }
        
        // Render Plotly charts
        {% for chart in charts %}
        Plotly.newPlot('{{ chart.id }}', {{ chart.data | tojson }}, {{ chart.layout | tojson }}, {responsive: true});
        {% endfor %}
        
        // Auto-select first column on page load
        document.addEventListener('DOMContentLoaded', function() {
            const selector = document.getElementById('columnSelector');
            if (selector && selector.options.length > 1) {
                selector.selectedIndex = 1; // Select first actual column (skip placeholder)
                showDistribution(selector.value);
            }
        });
    </script>
</body>
</html>
        '''
        
        # Render template
        from jinja2 import Environment, BaseLoader
        env = Environment(loader=BaseLoader())
        env.globals['format_datetime'] = format_datetime
        env.globals['truncate_text'] = truncate_text
        env.globals['zip'] = zip
        
        template = env.from_string(template_str)
        return template.render(analysis=analysis, charts=charts)


if __name__ == "__main__":
    # Test the evaluation system
    import pandas as pd
    
    # Create sample data
    data = {
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A'],
        'subcategory': ['X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'X'], 
        'value': [10, 20, 15, 25, 18, 12, 22, 16],
        'text': ['Sample text 1', 'Sample text 2', 'Sample text 3', 'Sample text 4',
                'Sample text 5', 'Sample text 6', 'Sample text 7', 'Sample text 8']
    }
    
    test_df = pd.DataFrame(data)
    
    # Generate report
    html_report = create_evaluation_report(test_df, output_path="test_evaluation_report.html")
    print("✅ Test evaluation report generated!")