import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import json

def initialize_session_state():
    """Initialize session state with completely empty configuration"""
    if 'column_config' not in st.session_state:
        st.session_state.column_config = {
            'key_columns': [],  # Start completely empty
            'amount_column': '',  # Start completely empty
            'additional_columns': []
        }
    
    if 'available_columns' not in st.session_state:
        st.session_state.available_columns = []
    
    if 'config_ready' not in st.session_state:
        st.session_state.config_ready = False

def read_file_headers_only(uploaded_file):
    """
    Read ONLY the headers (first row) from file - ultra fast
    Returns column list or None if failed
    """
    try:
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, nrows=0)  # Headers only
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';', nrows=0)  # Headers only
        
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, sheet_name=0, nrows=0)  # Headers only
        
        elif file_name.endswith('.tsv'):
            df = pd.read_csv(uploaded_file, sep='\t', nrows=0)  # Headers only
        
        else:
            df = pd.read_csv(uploaded_file, nrows=0)  # Headers only
        
        # Clean column names - remove dummy patterns
        dummy_patterns = ['unnamed:', 'column_', 'extra_column_', 'field_', 'col_', 'dummy', 'temp', 'placeholder', 'test']
        
        cleaned_columns = []
        for col in df.columns:
            col_str = str(col)
            col_lower = col_str.lower()
            is_dummy = any(pattern in col_lower for pattern in dummy_patterns)
            
            if not is_dummy and col_lower not in ['', 'nan', 'null', 'none']:
                cleaned_columns.append(col_str)
        
        return cleaned_columns
        
    except Exception as e:
        st.error(f"❌ Error reading file headers: {str(e)}")
        return None

def auto_detect_amount_column(columns):
    """Automatically detect the amount column from available columns"""
    amount_keywords = ['amount', 'value', 'cost', 'price', 'revenue', 'budget', 'total', 'sum', 'balance']
    currency_keywords = ['lc', 'usd', 'eur', 'gbp', 'currency']
    
    best_match = ''
    best_score = 0
    
    for col in columns:
        col_lower = str(col).lower()
        score = 0
        
        # High priority keywords
        if any(keyword in col_lower for keyword in amount_keywords):
            score += 3
        
        # Currency indicators
        if any(keyword in col_lower for keyword in currency_keywords):
            score += 2
        
        # Numeric-sounding names
        if any(word in col_lower for word in ['num', 'qty', 'quantity', 'count']):
            score += 1
        
        if score > best_score:
            best_score = score
            best_match = col
    
    return best_match

def configuration_from_file_upload():
    """Handle configuration file upload and extract columns"""
    st.sidebar.header("📤 Load Configuration from File")
    
    uploaded_config = st.sidebar.file_uploader(
        "Upload File to Extract Column Structure", 
        type=['xlsx', 'xls', 'csv', 'tsv'], 
        help="Upload any file to see its column structure",
        key="config_file"
    )
    
    if uploaded_config is not None:
        with st.spinner("⚡ Reading file headers..."):
            columns = read_file_headers_only(uploaded_config)
        
        if columns:
            st.session_state.available_columns = columns
            st.sidebar.success(f"✅ Found {len(columns)} columns")
            
            # Auto-detect amount column
            detected_amount = auto_detect_amount_column(columns)
            
            # All columns except amount column become key columns
            key_columns = [col for col in columns if col != detected_amount]
            
            # Clear any existing dummy configuration and set detected configuration
            st.session_state.column_config = {
                'key_columns': key_columns,  # All columns except amount
                'amount_column': detected_amount,  # Pre-select detected amount column
                'additional_columns': []  # No additional columns needed
            }
            
            st.sidebar.write("**Available Columns:**")
            for i, col in enumerate(columns):
                if col == detected_amount:
                    st.sidebar.write(f"{i+1}. `{col}` ⭐ **(Auto-detected as Amount Column)**")
                else:
                    st.sidebar.write(f"{i+1}. `{col}`")
            
            if detected_amount:
                st.sidebar.success(f"🎯 Auto-detected amount column: `{detected_amount}`")
            else:
                st.sidebar.info("ℹ️ No obvious amount column detected - please select manually")
            
            return True
    
    return False

def column_selection_panel():
    """Create column selection panel using actual file columns"""
    if not st.session_state.available_columns:
        st.sidebar.info("👆 Upload a file above to see its columns")
        return
    
    st.sidebar.header("🔧 Select Columns")
    
    available_cols = [''] + st.session_state.available_columns  # Empty option first
    
    # Amount Column Selection
    st.sidebar.write("**💰 Amount Column** (numerical values to compare)")
    
    # Get current amount column and find its index
    current_amount = st.session_state.column_config['amount_column']
    amount_index = 0
    if current_amount and current_amount in available_cols:
        amount_index = available_cols.index(current_amount)
    
    amount_col = st.sidebar.selectbox(
        "Select Amount Column",
        options=available_cols,
        index=amount_index,  # Use detected amount column index
        help="Choose the column containing numerical values",
        key="amount_select"
    )
    
    # Key Columns Selection
    st.sidebar.write("**🔑 Key Columns** (for matching records)")
    
    # Show current key columns
    for i in range(len(st.session_state.column_config['key_columns'])):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            selected = st.selectbox(
                f"Key Column {i+1}",
                options=available_cols,
                index=available_cols.index(st.session_state.column_config['key_columns'][i]) if st.session_state.column_config['key_columns'][i] in available_cols else 0,
                key=f"key_select_{i}"
            )
            if i < len(st.session_state.column_config['key_columns']):
                st.session_state.column_config['key_columns'][i] = selected
        with col2:
            if st.button("❌", key=f"remove_key_{i}"):
                st.session_state.column_config['key_columns'].pop(i)
                st.rerun()
    
    # Add key column button
    if st.sidebar.button("➕ Add Key Column"):
        st.session_state.column_config['key_columns'].append('')
        st.rerun()
    
    # Additional Columns Selection
    st.sidebar.write("**📊 Additional Columns** (optional)")
    
    # Show current additional columns
    for i in range(len(st.session_state.column_config['additional_columns'])):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            selected = st.selectbox(
                f"Additional Column {i+1}",
                options=available_cols,
                index=available_cols.index(st.session_state.column_config['additional_columns'][i]) if st.session_state.column_config['additional_columns'][i] in available_cols else 0,
                key=f"add_select_{i}"
            )
            if i < len(st.session_state.column_config['additional_columns']):
                st.session_state.column_config['additional_columns'][i] = selected
        with col2:
            if st.button("❌", key=f"remove_add_{i}"):
                st.session_state.column_config['additional_columns'].pop(i)
                st.rerun()
    
    # Add additional column button
    if st.sidebar.button("➕ Add Additional Column"):
        st.session_state.column_config['additional_columns'].append('')
        st.rerun()
    
    # Update amount column and automatically adjust key columns
    if amount_col != st.session_state.column_config['amount_column']:
        st.session_state.column_config['amount_column'] = amount_col
        
        # Automatically set all non-amount columns as key columns
        if amount_col:
            st.session_state.column_config['key_columns'] = [col for col in st.session_state.available_columns if col != amount_col]
        else:
            st.session_state.column_config['key_columns'] = st.session_state.available_columns.copy()
    
    # Configuration Summary
    st.sidebar.write("---")
    st.sidebar.write("**📋 Current Selection:**")
    if amount_col:
        st.sidebar.write(f"**💰 Amount:** `{amount_col}`")
    
    key_cols_clean = [col for col in st.session_state.column_config['key_columns'] if col]
    if key_cols_clean:
        st.sidebar.write(f"**🔑 Keys:** `{', '.join(key_cols_clean)}`")
        st.sidebar.info(f"ℹ️ All {len(key_cols_clean)} non-amount columns are automatically set as key columns")
    
    add_cols_clean = [col for col in st.session_state.column_config['additional_columns'] if col]
    if add_cols_clean:
        st.sidebar.write(f"**📊 Additional:** `{', '.join(add_cols_clean)}`")

def read_full_file(uploaded_file):
    """Read complete file for analysis"""
    try:
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';')
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, sheet_name=0)
        elif file_name.endswith('.tsv'):
            df = pd.read_csv(uploaded_file, sep='\t')
        else:
            df = pd.read_csv(uploaded_file)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None

def validate_configuration():
    """Check if configuration is valid"""
    amount_col = st.session_state.column_config['amount_column']
    key_cols = [col for col in st.session_state.column_config['key_columns'] if col]
    
    if not amount_col:
        return False, "Please select an amount column"
    
    if not key_cols:
        return False, "Please select at least one key column"
    
    return True, "Configuration is valid"

def get_unique_records(df1, df2, file1_name, file2_name):
    """Get records that exist only in one file but not the other"""
    amount_column = st.session_state.column_config['amount_column']
    key_columns = [col for col in st.session_state.column_config['key_columns'] if col]
    
    # Create comparison keys
    df1_comp = df1.copy()
    df2_comp = df2.copy()
    
    # Create composite keys
    key_parts_1 = [df1_comp[col].astype(str) for col in key_columns if col in df1_comp.columns]
    key_parts_2 = [df2_comp[col].astype(str) for col in key_columns if col in df2_comp.columns]
    
    if key_parts_1:
        df1_comp['Comparison_Key'] = key_parts_1[0]
        for part in key_parts_1[1:]:
            df1_comp['Comparison_Key'] = df1_comp['Comparison_Key'] + '|' + part
    
    if key_parts_2:
        df2_comp['Comparison_Key'] = key_parts_2[0]
        for part in key_parts_2[1:]:
            df2_comp['Comparison_Key'] = df2_comp['Comparison_Key'] + '|' + part
    
    # Find unique keys
    keys_file1 = set(df1_comp['Comparison_Key'])
    keys_file2 = set(df2_comp['Comparison_Key'])
    
    # Records only in file1
    only_in_file1_keys = keys_file1 - keys_file2
    only_in_file1 = df1_comp[df1_comp['Comparison_Key'].isin(only_in_file1_keys)].copy()
    
    # Records only in file2
    only_in_file2_keys = keys_file2 - keys_file1
    only_in_file2 = df2_comp[df2_comp['Comparison_Key'].isin(only_in_file2_keys)].copy()
    
    return only_in_file1, only_in_file2

def calculate_variance_reconciliation(comparison_df, target_percentage=85):
    """Calculate variance reconciliation to identify minimum records explaining target % of variance"""
    # Filter out records with no difference
    variance_records = comparison_df[comparison_df['Abs_Difference'] > 0].copy()
    
    if len(variance_records) == 0:
        return pd.DataFrame(), 0, 0, 0
    
    # Sort by absolute difference (largest variances first)
    variance_records = variance_records.sort_values('Abs_Difference', ascending=False)
    
    # Calculate cumulative variance
    total_variance = variance_records['Abs_Difference'].sum()
    variance_records['Cumulative_Variance'] = variance_records['Abs_Difference'].cumsum()
    variance_records['Cumulative_Percentage'] = (variance_records['Cumulative_Variance'] / total_variance) * 100
    
    # Find records that explain target percentage of variance
    target_records = variance_records[variance_records['Cumulative_Percentage'] <= target_percentage].copy()
    
    # If no records meet the target, take at least the top record
    if len(target_records) == 0:
        target_records = variance_records.head(1).copy()
    
    # Add variance contribution for each record
    target_records['Variance_Contribution'] = (target_records['Abs_Difference'] / total_variance) * 100
    
    actual_percentage = target_records['Cumulative_Percentage'].iloc[-1] if len(target_records) > 0 else 0
    records_count = len(target_records)
    
    return target_records, actual_percentage, records_count, total_variance

def create_visualizations(comparison_df, file1_name, file2_name):
    """Create interactive visualizations for the comparison"""
    key_columns = [col for col in st.session_state.column_config['key_columns'] if col]
    
    # Use first key column for grouping
    group_column = key_columns[0] if key_columns else 'Status'
    
    # 1. Summary by first key column
    if group_column in comparison_df.columns:
        group_summary = comparison_df.groupby(group_column).agg({
            f'Amount_{file1_name}': 'sum',
            f'Amount_{file2_name}': 'sum',
            'Difference': 'sum'
        }).reset_index()
        
        fig_group = go.Figure()
        fig_group.add_trace(go.Bar(name=file1_name, x=group_summary[group_column], y=group_summary[f'Amount_{file1_name}']))
        fig_group.add_trace(go.Bar(name=file2_name, x=group_summary[group_column], y=group_summary[f'Amount_{file2_name}']))
        fig_group.update_layout(title=f'Total Amounts by {group_column}', barmode='group')
    else:
        # Fallback: Status summary
        status_summary = comparison_df.groupby('Status').agg({
            f'Amount_{file1_name}': 'sum',
            f'Amount_{file2_name}': 'sum',
            'Difference': 'sum'
        }).reset_index()
        
        fig_group = go.Figure()
        fig_group.add_trace(go.Bar(name=file1_name, x=status_summary['Status'], y=status_summary[f'Amount_{file1_name}']))
        fig_group.add_trace(go.Bar(name=file2_name, x=status_summary['Status'], y=status_summary[f'Amount_{file2_name}']))
        fig_group.update_layout(title='Total Amounts by Status', barmode='group')
    
    # 2. Difference Distribution
    fig_diff = px.histogram(comparison_df[comparison_df['Difference'] != 0], 
                           x='Difference', nbins=50, 
                           title='Distribution of Differences (Non-Zero Only)')
    
    # 3. Status Pie Chart
    status_counts = comparison_df['Status'].value_counts()
    fig_status = px.pie(values=status_counts.values, names=status_counts.index, 
                       title='Records by Change Status')
    
    return fig_group, fig_diff, fig_status

def perform_comparison(df1, df2, file1_name, file2_name):
    """Compare two DataFrames using selected configuration"""
    amount_column = st.session_state.column_config['amount_column']
    key_columns = [col for col in st.session_state.column_config['key_columns'] if col]
    additional_columns = [col for col in st.session_state.column_config['additional_columns'] if col]
    
    # Create comparison keys
    df1_comp = df1.copy()
    df2_comp = df2.copy()
    
    # Create composite key
    key_parts_1 = [df1_comp[col].astype(str) for col in key_columns if col in df1_comp.columns]
    key_parts_2 = [df2_comp[col].astype(str) for col in key_columns if col in df2_comp.columns]
    
    if key_parts_1:
        df1_comp['Comparison_Key'] = key_parts_1[0]
        for part in key_parts_1[1:]:
            df1_comp['Comparison_Key'] = df1_comp['Comparison_Key'] + '|' + part
    
    if key_parts_2:
        df2_comp['Comparison_Key'] = key_parts_2[0]
        for part in key_parts_2[1:]:
            df2_comp['Comparison_Key'] = df2_comp['Comparison_Key'] + '|' + part
    
    # Prepare for merge
    merge_cols_1 = ['Comparison_Key'] + key_columns + additional_columns + [amount_column]
    merge_cols_1 = [col for col in merge_cols_1 if col in df1_comp.columns]
    
    merge_cols_2 = ['Comparison_Key', amount_column]
    merge_cols_2 = [col for col in merge_cols_2 if col in df2_comp.columns]
    
    df1_merge = df1_comp[merge_cols_1].copy()
    df2_merge = df2_comp[merge_cols_2].copy()
    
    # Rename amount columns
    df1_merge = df1_merge.rename(columns={amount_column: f'Amount_{file1_name}'})
    df2_merge = df2_merge.rename(columns={amount_column: f'Amount_{file2_name}'})
    
    # Merge
    comparison_df = pd.merge(df1_merge, df2_merge, on='Comparison_Key', how='outer')
    
    # Fill missing values
    comparison_df[f'Amount_{file1_name}'] = comparison_df[f'Amount_{file1_name}'].fillna(0)
    comparison_df[f'Amount_{file2_name}'] = comparison_df[f'Amount_{file2_name}'].fillna(0)
    
    # Calculate differences
    comparison_df['Difference'] = comparison_df[f'Amount_{file2_name}'] - comparison_df[f'Amount_{file1_name}']
    comparison_df['Abs_Difference'] = comparison_df['Difference'].abs()
    
    # Calculate percentage change (handle division by zero)
    comparison_df['Percentage_Change'] = np.where(
        comparison_df[f'Amount_{file1_name}'] != 0,
        (comparison_df['Difference'] / comparison_df[f'Amount_{file1_name}']) * 100,
        np.where(comparison_df[f'Amount_{file2_name}'] != 0, np.inf, 0)
    )
    
    # Add status
    comparison_df['Status'] = np.where(
        comparison_df['Difference'] == 0, 'No Change',
        np.where(comparison_df['Difference'] > 0, 'Increase', 'Decrease')
    )
    
    # Add record source information
    comparison_df['Record_Source'] = np.where(
        (comparison_df[f'Amount_{file1_name}'] != 0) & (comparison_df[f'Amount_{file2_name}'] != 0), 'Both Files',
        np.where(comparison_df[f'Amount_{file1_name}'] != 0, f'Only {file1_name}', f'Only {file2_name}')
    )
    
    # Fix data types for better compatibility
    for col in key_columns + additional_columns:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].astype(str)
    
    return comparison_df

def create_summary_metrics(comparison_df, file1_name, file2_name):
    """Create summary metrics for the comparison"""
    total_records = len(comparison_df)
    
    # Financial summaries
    total_file1 = comparison_df[f'Amount_{file1_name}'].sum()
    total_file2 = comparison_df[f'Amount_{file2_name}'].sum()
    total_difference = comparison_df['Difference'].sum()
    
    # Record counts by status
    no_change_count = len(comparison_df[comparison_df['Status'] == 'No Change'])
    increase_count = len(comparison_df[comparison_df['Status'] == 'Increase'])
    decrease_count = len(comparison_df[comparison_df['Status'] == 'Decrease'])
    
    # Records by source
    both_files_count = len(comparison_df[comparison_df['Record_Source'] == 'Both Files'])
    only_file1_count = len(comparison_df[comparison_df['Record_Source'] == f'Only {file1_name}'])
    only_file2_count = len(comparison_df[comparison_df['Record_Source'] == f'Only {file2_name}'])
    
    return {
        'total_records': total_records,
        'total_file1': total_file1,
        'total_file2': total_file2,
        'total_difference': total_difference,
        'no_change_count': no_change_count,
        'increase_count': increase_count,
        'decrease_count': decrease_count,
        'both_files_count': both_files_count,
        'only_file1_count': only_file1_count,
        'only_file2_count': only_file2_count
    }

def main():
    st.set_page_config(
        page_title="Flexible Comparison Dashboard", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    st.title("🔧 Flexible Comparison Dashboard")
    st.markdown("**Upload a file to extract its column structure, then configure and compare**")
    
    # Configuration from file upload
    config_loaded = configuration_from_file_upload()
    
    # Column selection panel (only shows if columns are available)
    column_selection_panel()
    
    # Main file upload area
    st.header("📁 File Comparison")
    
    # Only show file upload if configuration is available
    if st.session_state.available_columns:
        
        # Validate configuration
        is_valid, message = validate_configuration()
        
        if not is_valid:
            st.warning(f"⚠️ Configuration incomplete: {message}")
            st.info("👈 Please complete your column selection in the sidebar")
            return
        
        # File upload for comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 First File")
            file1 = st.file_uploader("Upload First File", type=['xlsx', 'xls', 'csv', 'tsv'], key="file1")
            file1_name = st.text_input("First File Label", value="File_1", key="name1")
        
        with col2:
            st.subheader("📄 Second File")
            file2 = st.file_uploader("Upload Second File", type=['xlsx', 'xls', 'csv', 'tsv'], key="file2")
            file2_name = st.text_input("Second File Label", value="File_2", key="name2")
        
        # Process files if both uploaded
        if file1 is not None and file2 is not None:
            
            with st.spinner("Loading files for comparison..."):
                df1 = read_full_file(file1)
                df2 = read_full_file(file2)
            
            if df1 is not None and df2 is not None:
                
                # Validate columns exist in files
                amount_col = st.session_state.column_config['amount_column']
                key_cols = [col for col in st.session_state.column_config['key_columns'] if col]
                
                missing_in_file1 = [col for col in [amount_col] + key_cols if col not in df1.columns]
                missing_in_file2 = [col for col in [amount_col] + key_cols if col not in df2.columns]
                
                if missing_in_file1:
                    st.error(f"❌ File 1 missing columns: {missing_in_file1}")
                    return
                
                if missing_in_file2:
                    st.error(f"❌ File 2 missing columns: {missing_in_file2}")
                    return
                
                # Convert amount columns to numeric
                df1[amount_col] = pd.to_numeric(df1[amount_col], errors='coerce').fillna(0)
                df2[amount_col] = pd.to_numeric(df2[amount_col], errors='coerce').fillna(0)
                
                # Perform comparison and get all analysis data
                with st.spinner("Performing comprehensive analysis..."):
                    comparison_df = perform_comparison(df1, df2, file1_name, file2_name)
                    summary_metrics = create_summary_metrics(comparison_df, file1_name, file2_name)
                    only_in_file1, only_in_file2 = get_unique_records(df1, df2, file1_name, file2_name)
                
                # Display summary metrics
                st.header("📈 Summary Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", f"{summary_metrics['total_records']:,}")
                    st.metric("Records in Both Files", f"{summary_metrics['both_files_count']:,}")
                
                with col2:
                    st.metric(f"Total {file1_name}", f"{summary_metrics['total_file1']:,.2f}")
                    st.metric(f"Only in {file1_name}", f"{summary_metrics['only_file1_count']:,}")
                
                with col3:
                    st.metric(f"Total {file2_name}", f"{summary_metrics['total_file2']:,.2f}")
                    st.metric(f"Only in {file2_name}", f"{summary_metrics['only_file2_count']:,}")
                
                with col4:
                    st.metric("Total Difference", f"{summary_metrics['total_difference']:,.2f}")
                    percentage_change = ((summary_metrics['total_file2'] - summary_metrics['total_file1']) / 
                                       summary_metrics['total_file1'] * 100) if summary_metrics['total_file1'] != 0 else 0
                    st.metric("Overall % Change", f"{percentage_change:.2f}%")
                
                # Change status breakdown
                st.subheader("Change Status Breakdown")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("No Change", f"{summary_metrics['no_change_count']:,}")
                with col2:
                    st.metric("Increases", f"{summary_metrics['increase_count']:,}")
                with col3:
                    st.metric("Decreases", f"{summary_metrics['decrease_count']:,}")
                
                # Visualizations
                st.header("📊 Visual Analysis")
                
                with st.spinner("Creating visualizations..."):
                    fig_group, fig_diff, fig_status = create_visualizations(comparison_df, file1_name, file2_name)
                
                # Display charts in tabs
                tab1, tab2, tab3 = st.tabs(["Group Comparison", "Difference Distribution", "Change Status"])
                
                with tab1:
                    st.plotly_chart(fig_group, width='stretch')
                
                with tab2:
                    st.plotly_chart(fig_diff, width='stretch')
                
                with tab3:
                    st.plotly_chart(fig_status, width='stretch')
                
                # Advanced Analysis Views
                st.header("🔍 Detailed Analysis Views")
                
                # Variance reconciliation
                st.sidebar.header("⚖️ Variance Reconciliation")
                target_percentage = st.sidebar.slider(
                    "Target Variance Coverage (%)", 
                    min_value=50, 
                    max_value=99, 
                    value=85, 
                    step=5,
                    help="Percentage of total variance to explain with minimum records"
                )
                
                variance_records, actual_percentage, records_count, total_variance = calculate_variance_reconciliation(
                    comparison_df, target_percentage
                )
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs([
                    f"📊 Complete Comparison ({len(comparison_df):,} records)",
                    f"⚖️ Variance Reconciliation ({records_count:,} records)",
                    f"📋 Only in {file1_name} ({len(only_in_file1):,} records)",
                    f"📋 Only in {file2_name} ({len(only_in_file2):,} records)"
                ])
                
                with tab1:
                    st.subheader("Complete Comparison View")
                    
                    # Format display columns
                    display_cols = key_cols + [f'Amount_{file1_name}', f'Amount_{file2_name}', 'Difference', 'Percentage_Change', 'Status']
                    display_cols = [col for col in display_cols if col in comparison_df.columns]
                    
                    # Format for display
                    formatted_df = comparison_df[display_cols].copy()
                    for col in [f'Amount_{file1_name}', f'Amount_{file2_name}', 'Difference']:
                        if col in formatted_df.columns:
                            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.2f}")
                    if 'Percentage_Change' in formatted_df.columns:
                        formatted_df['Percentage_Change'] = formatted_df['Percentage_Change'].apply(
                            lambda x: f"{x:.2f}%" if x != np.inf else "New Record"
                        )
                    
                    st.dataframe(formatted_df, width='stretch')
                
                with tab2:
                    st.subheader("⚖️ Variance Reconciliation Analysis")
                    
                    if len(variance_records) > 0:
                        st.write(f"**Key Insight**: {records_count:,} records explain {actual_percentage:.1f}% of total variance")
                        
                        # Reconciliation metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Records Analyzed", f"{records_count:,}")
                        with col2:
                            st.metric("Variance Explained", f"{actual_percentage:.1f}%")
                        with col3:
                            st.metric("Total Variance", f"{total_variance:,.2f}")
                        with col4:
                            efficiency = (actual_percentage / records_count) if records_count > 0 else 0
                            st.metric("Efficiency Ratio", f"{efficiency:.2f}%/record")
                        
                        # Top contributors chart
                        st.subheader("📈 Top Variance Contributors")
                        
                        top_10 = variance_records.head(10)
                        
                        # Create labels using key columns
                        labels = []
                        for _, row in top_10.iterrows():
                            label_parts = []
                            for col in key_cols:
                                if col in row and pd.notna(row[col]):
                                    label_parts.append(str(row[col]))
                            label_parts.append(f"{row['Abs_Difference']:,.0f}")
                            labels.append("<br>".join(label_parts))
                        
                        fig_waterfall = go.Figure()
                        fig_waterfall.add_trace(go.Bar(
                            x=list(range(len(top_10))),
                            y=top_10['Abs_Difference'],
                            text=labels,
                            textposition='auto',
                            name='Variance Contribution'
                        ))
                        
                        fig_waterfall.update_layout(
                            title=f"Top 10 Variance Contributors (Total: {top_10['Abs_Difference'].sum():,.2f})",
                            xaxis_title="Record Rank",
                            yaxis_title="Absolute Variance",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_waterfall, width='stretch')
                        
                        # Detailed reconciliation table
                        st.subheader("🔍 Detailed Reconciliation Records")
                        
                        var_display_columns = key_cols + [
                            f'Amount_{file1_name}', f'Amount_{file2_name}', 'Difference', 
                            'Abs_Difference', 'Variance_Contribution', 'Cumulative_Percentage'
                        ]
                        var_display_columns = [col for col in var_display_columns if col in variance_records.columns]
                        
                        # Format for display
                        formatted_var = variance_records[var_display_columns].copy()
                        for col in [f'Amount_{file1_name}', f'Amount_{file2_name}', 'Difference', 'Abs_Difference']:
                            if col in formatted_var.columns:
                                formatted_var[col] = formatted_var[col].apply(lambda x: f"{x:,.2f}")
                        if 'Variance_Contribution' in formatted_var.columns:
                            formatted_var['Variance_Contribution'] = formatted_var['Variance_Contribution'].apply(lambda x: f"{x:.2f}%")
                        if 'Cumulative_Percentage' in formatted_var.columns:
                            formatted_var['Cumulative_Percentage'] = formatted_var['Cumulative_Percentage'].apply(lambda x: f"{x:.2f}%")
                        
                        st.dataframe(formatted_var, width='stretch')
                    
                    else:
                        st.info("No variance records found - all amounts are identical between files")
                
                with tab3:
                    st.subheader(f"Records Only in {file1_name}")
                    st.write(f"Shows {len(only_in_file1):,} records that exist only in {file1_name} but not in {file2_name}")
                    
                    if len(only_in_file1) > 0:
                        # Display columns for unique records
                        unique_display_columns = key_cols + [amount_col]
                        unique_display_columns = [col for col in unique_display_columns if col in only_in_file1.columns]
                        
                        # Format for display
                        formatted_f1 = only_in_file1[unique_display_columns].copy()
                        if amount_col in formatted_f1.columns:
                            formatted_f1[amount_col] = formatted_f1[amount_col].apply(lambda x: f"{x:,.2f}")
                        
                        st.dataframe(formatted_f1, width='stretch')
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Records", f"{len(only_in_file1):,}")
                        with col2:
                            if amount_col in only_in_file1.columns:
                                total_amount_f1 = only_in_file1[amount_col].sum()
                                st.metric("Total Amount", f"{total_amount_f1:,.2f}")
                        with col3:
                            if amount_col in only_in_file1.columns:
                                avg_amount_f1 = only_in_file1[amount_col].mean() if len(only_in_file1) > 0 else 0
                                st.metric("Average Amount", f"{avg_amount_f1:,.2f}")
                    
                    else:
                        st.info(f"No unique records found in {file1_name}")
                
                with tab4:
                    st.subheader(f"Records Only in {file2_name}")
                    st.write(f"Shows {len(only_in_file2):,} records that exist only in {file2_name} but not in {file1_name}")
                    
                    if len(only_in_file2) > 0:
                        # Display columns for unique records
                        unique_display_columns = key_cols + [amount_col]
                        unique_display_columns = [col for col in unique_display_columns if col in only_in_file2.columns]
                        
                        # Format for display
                        formatted_f2 = only_in_file2[unique_display_columns].copy()
                        if amount_col in formatted_f2.columns:
                            formatted_f2[amount_col] = formatted_f2[amount_col].apply(lambda x: f"{x:,.2f}")
                        
                        st.dataframe(formatted_f2, width='stretch')
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Records", f"{len(only_in_file2):,}")
                        with col2:
                            if amount_col in only_in_file2.columns:
                                total_amount_f2 = only_in_file2[amount_col].sum()
                                st.metric("Total Amount", f"{total_amount_f2:,.2f}")
                        with col3:
                            if amount_col in only_in_file2.columns:
                                avg_amount_f2 = only_in_file2[amount_col].mean() if len(only_in_file2) > 0 else 0
                                st.metric("Average Amount", f"{avg_amount_f2:,.2f}")
                    
                    else:
                        st.info(f"No unique records found in {file2_name}")
                
                # Enhanced Download Options
                st.header("💾 Download Results")
                
                # Prepare comprehensive download data
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    
                    # Sheet 1: Complete Comparison
                    display_cols = key_cols + [f'Amount_{file1_name}', f'Amount_{file2_name}', 'Difference', 'Percentage_Change', 'Status']
                    display_cols = [col for col in display_cols if col in comparison_df.columns]
                    comparison_df[display_cols].to_excel(writer, sheet_name='Complete_Comparison', index=False)
                    
                    # Sheet 2: Variance Reconciliation
                    if len(variance_records) > 0:
                        var_columns = key_cols + [
                            f'Amount_{file1_name}', f'Amount_{file2_name}', 'Difference', 
                            'Abs_Difference', 'Variance_Contribution', 'Cumulative_Percentage'
                        ]
                        var_columns = [col for col in var_columns if col in variance_records.columns]
                        variance_records[var_columns].to_excel(writer, sheet_name='Variance_Reconciliation', index=False)
                    
                    # Sheet 3: Only in File1
                    if len(only_in_file1) > 0:
                        unique_columns = key_cols + [amount_col]
                        unique_columns = [col for col in unique_columns if col in only_in_file1.columns]
                        only_in_file1[unique_columns].to_excel(writer, sheet_name=f'Only_in_{file1_name}', index=False)
                    
                    # Sheet 4: Only in File2
                    if len(only_in_file2) > 0:
                        unique_columns = key_cols + [amount_col]
                        unique_columns = [col for col in unique_columns if col in only_in_file2.columns]
                        only_in_file2[unique_columns].to_excel(writer, sheet_name=f'Only_in_{file2_name}', index=False)
                    
                    # Sheet 5: Summary Statistics
                    summary_data = pd.DataFrame([
                        ['Metric', 'Value'],
                        ['Total Records', summary_metrics['total_records']],
                        [f'Total {file1_name}', summary_metrics['total_file1']],
                        [f'Total {file2_name}', summary_metrics['total_file2']],
                        ['Total Difference', summary_metrics['total_difference']],
                        ['Records with No Change', summary_metrics['no_change_count']],
                        ['Records with Increases', summary_metrics['increase_count']],
                        ['Records with Decreases', summary_metrics['decrease_count']],
                        ['Records in Both Files', summary_metrics['both_files_count']],
                        [f'Records Only in {file1_name}', summary_metrics['only_file1_count']],
                        [f'Records Only in {file2_name}', summary_metrics['only_file2_count']],
                        ['Variance Records Analyzed', records_count],
                        ['Variance Coverage Achieved', f"{actual_percentage:.1f}%"],
                        ['Total Variance Amount', total_variance]
                    ])
                    summary_data.to_excel(writer, sheet_name='Summary_Statistics', index=False, header=False)
                    
                    # Sheet 6: Column Configuration
                    config_data = pd.DataFrame([
                        ['Configuration', 'Value'],
                        ['Amount Column', st.session_state.column_config['amount_column']],
                        ['Key Columns', ', '.join(key_cols)],
                        ['Additional Columns', ', '.join([col for col in st.session_state.column_config['additional_columns'] if col])]
                    ])
                    config_data.to_excel(writer, sheet_name='Column_Configuration', index=False, header=False)
                
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Download Complete Analysis (Excel)",
                    data=excel_data,
                    file_name=f"flexible_comparison_{file1_name}_vs_{file2_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    else:
        st.info("👆 Please upload a configuration file in the sidebar to get started")
        
        st.subheader("🔧 How to Use")
        st.write("1. **Upload Configuration File**: Use sidebar to upload any Excel/CSV file")
        st.write("2. **Select Columns**: Choose amount column and key columns from your file")
        st.write("3. **Upload Comparison Files**: Upload two files with the same structure")
        st.write("4. **Analyze Results**: View comparison and download results")

if __name__ == "__main__":
    main()
