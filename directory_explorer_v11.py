# /// script
# [tool.marimo.experimental]
# multi_column = true
# ///

# directory_explorer_v11.py - Enhanced with deployment path support
# v11: Added support for Railway/cloud deployment with simplified paths

import marimo

__generated_with = "0.14.6"
app = marimo.App(
    width="columns",
    layout_file="layouts/directory_explorer_v10.grid.json",
)


@app.cell(column=0)
def _():
    import marimo as mo
    import os
    from typing import List, Optional, Dict, Any, Tuple
    from dataclasses import dataclass

    import pickle
    import pandas as pd
    import polars as pl
    from pathlib import Path
    import numpy as np

    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.colors as pc
    from plotly_resampler import FigureResampler as FigureWidgetResampler
    return (
        Any,
        Dict,
        FigureWidgetResampler,
        List,
        Optional,
        dataclass,
        go,
        mo,
        os,
        pc,
        pd,
        pickle,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Column_A: Print Directory Structure Viewer""")
    return


@app.cell(hide_code=True)
def _(os):
    # v11: Simple deployment detection
    def is_deployment_environment():
        """Check if running in deployment environment (Railway, etc.)"""
        # Check for common deployment indicators
        return any([
            os.environ.get('RAILWAY_ENVIRONMENT'),
            os.environ.get('PORT'),  # Common in cloud deployments
            os.path.exists('./Procfile'),  # Deployment indicator
            # Check if we're NOT in a typical Windows/WSL environment
            (not os.path.exists('C:\\') and not os.path.exists('/mnt/c/'))
        ])
    
    # Machine Environment Configuration
    MACHINE_ENVIRONMENTS = {
        "DEPLOYMENT": {
            "label": "Railway/Cloud Deployment",
            "start_path": os.path.abspath('.'),  # Current directory in deployment
            "description": "Cloud deployment environment"
        },
        "WFH": {
            "label": "Work From Home",
            "start_path": r"C:\Users\Michael\Dropbox (Personal)",
            "description": "Home office setup"
        },
        "PROCESS_STATION": {
            "label": "Process Station / Automation_1 / Automation_2", 
            "start_path": r"C:\Users\dimas\Dropbox",
            "description": "Main process automation stations"
        },
        "ANALOG_STATION": {
            "label": "ANALOG-STATION-1 / ANALOG-STATION-2",
            "start_path": r"C:\Users\Michael Di Mascio\Dropbox",
            "description": "Analog testing stations"
        },
        "SOCKET_STATION": {
            "label": "SOCKET-STATION",
            "start_path": r"C:\Users\rktpo\Dropbox", 
            "description": "Socket testing station"
        },
        "OTHER": {
            "label": "Other / Custom Path",
            "start_path": None,  # Will be manually entered
            "description": "Custom or unknown environment"
        }
    }

    def detect_current_environment() -> str:
        """Auto-detect current machine environment based on existing paths"""
        # v11: Check deployment environment first
        if is_deployment_environment():
            return "DEPLOYMENT"
            
        for env_key, env_config in MACHINE_ENVIRONMENTS.items():
            if env_key in ["OTHER", "DEPLOYMENT"]:
                continue
            if env_config["start_path"] and os.path.exists(env_config["start_path"]):
                return env_key
        return "OTHER"

    def resolve_environment_paths(selected_env: str, custom_path: str = "") -> tuple:
        """Resolve start paths based on environment selection"""

        if selected_env == "OTHER":
            if not custom_path:
                raise ValueError("Custom path required for 'Other' environment")
            start_path = custom_path.strip().rstrip("\\")
        else:
            env_config = MACHINE_ENVIRONMENTS[selected_env]
            start_path = env_config["start_path"]

        # Validate path exists
        if not os.path.exists(start_path):
            raise FileNotFoundError(f"Path does not exist: {start_path}")

        # Construct project paths using os.path.join for better path handling
        sequencer_path = os.path.join(start_path, 
            "USDCT1_PA", "Parent_Dir", "Data_to_organize", 
            "Lab_Instrument_Doc", "TC2_software", "Efficiency_v2", ".sequencer")

        sps_gui_path = os.path.join(start_path,
            "USDCT1_PA", "Parent_Dir", "Data_to_organize",
            "Lab_Instrument_Doc", "TC2_software", "sps_gui")

        return start_path, sequencer_path, sps_gui_path

    return (
        MACHINE_ENVIRONMENTS,
        detect_current_environment,
        is_deployment_environment,
        resolve_environment_paths,
    )


@app.cell(hide_code=True)
def _(MACHINE_ENVIRONMENTS, detect_current_environment, mo):
    # Environment selector dropdown
    detected_env = detect_current_environment()

    machine_environment_selector = mo.ui.dropdown(
        options=list(MACHINE_ENVIRONMENTS.keys()),
        value=detected_env,
        label="Select Machine Environment"
    )

    # Custom path input (conditional)
    custom_path_input = mo.ui.text(
        placeholder="Enter custom Dropbox path (e.g., C:\\Users\\username\\Dropbox)",
        label="Custom Path",
        value=""
    )

    # Auto-detection status
    env_label = MACHINE_ENVIRONMENTS[detected_env]["label"]
    detection_status = mo.md(f"🤖 **Auto-detected:** {env_label}")

    return custom_path_input, detection_status, machine_environment_selector


@app.cell(hide_code=True)
def _(
    MACHINE_ENVIRONMENTS,
    custom_path_input,
    detection_status,
    machine_environment_selector,
    mo,
):
    # Environment selection UI and path resolution
    selected_env = machine_environment_selector.value
    env_config = MACHINE_ENVIRONMENTS[selected_env]

    # Build environment info display
    env_info_lines = [
        f"## Selected Environment: {env_config['label']}",
        f"**Description:** {env_config['description']}",
        ""
    ]

    # Show custom path input only for "OTHER" environment
    if selected_env == "OTHER":
        environment_ui = mo.vstack([
            detection_status,
            machine_environment_selector,
            custom_path_input,
            mo.md("\n".join(env_info_lines))
        ])
    else:
        env_info_lines.append(f"**Path:** `{env_config['start_path']}`")
        environment_ui = mo.vstack([
            detection_status,
            machine_environment_selector,
            mo.md("\n".join(env_info_lines))
        ])

    return environment_ui, selected_env


@app.cell(hide_code=True)
def _(custom_path_input, mo, resolve_environment_paths, selected_env):
    # Path resolution with error handling
    try:
        resolved_start_path, global_sequencer_path, global_sps_gui_path = resolve_environment_paths(
            selected_env, 
            custom_path_input.value if selected_env == "OTHER" else ""
        )

        path_status = mo.md(f"""
        ### ✅ Path Resolution Successful

        **Base Path:** `{resolved_start_path}`  
        **Sequencer Path:** `{global_sequencer_path}`  
        **SPS GUI Path:** `{global_sps_gui_path}`
        """)

        print(f"Environment: {selected_env}")
        print(f"Resolved start path: {resolved_start_path}")
        print(f"global_sequencer_path: {global_sequencer_path}")
        print(f"global_sps_gui_path: {global_sps_gui_path}")

    except (ValueError, FileNotFoundError) as e:
        # Set default fallback paths to prevent downstream errors
        resolved_start_path = ""
        global_sequencer_path = ""
        global_sps_gui_path = ""

        path_status = mo.md(f"""
        ### ❌ Path Resolution Error

        **Error:** {str(e)}

        Please check your environment selection or provide a valid custom path.
        """)

        print(f"Path resolution error: {e}")

    return (
        global_sequencer_path,
        global_sps_gui_path,
        path_status,
        resolved_start_path,
    )


@app.cell(hide_code=True)
def _(List, Optional, os):
    def print_directory_structure(
        startpath: str, 
        prefix: str = '',
        exclude_dirs: Optional[List[str]] = None,
        collapse_dirs: Optional[List[str]] = None
    ) -> List[str]:
        """
        Returns directory structure as list of strings for display.
        """
        if exclude_dirs is None:
            exclude_dirs = ['__pycache__']
        if collapse_dirs is None:
            collapse_dirs = ['icon']

        lines = []
        files = []

        if prefix:
            prefix = prefix.replace('├──', '│   ').replace('└──', '    ')

        try:
            entries = sorted(os.listdir(startpath))
        except (PermissionError, FileNotFoundError) as e:
            return [f"{prefix}├── [Error: {e}]"]

        for entry in entries:
            path = os.path.join(startpath, entry)

            if os.path.isdir(path):
                if entry in exclude_dirs:
                    continue

                if entry in collapse_dirs:
                    try:
                        item_count = len(os.listdir(path))
                        lines.append(f"{prefix}├── {entry}/ ({item_count} items)")
                    except:
                        lines.append(f"{prefix}├── {entry}/ [Access Denied]")
                    continue

                lines.append(f"{prefix}├── {entry}/")
                sublines = print_directory_structure(
                    path, 
                    prefix + '│   ',
                    exclude_dirs,
                    collapse_dirs
                )
                lines.extend(sublines)
            else:
                files.append(entry)

        for i, file in enumerate(files):
            branch = '└── ' if i == len(files) - 1 else '├── '
            lines.append(f"{prefix}{branch}{file}")

        return lines

    return (print_directory_structure,)


@app.cell(hide_code=True)
def _(global_sequencer_path, global_sps_gui_path, os):
    # Define the package directory global_package_paths
    global_package_paths = {
        "sequencer": global_sequencer_path,
        "sps_gui": global_sps_gui_path
    }

    # Check if global_package_paths exist
    for name, path in global_package_paths.items():
        if path and os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (not found or invalid)")

    return (global_package_paths,)


@app.cell(hide_code=True)
def _(mo):
    # Simple UI for directory selection
    package_selector = mo.ui.dropdown(
        options=["sequencer", "sps_gui", "both"],
        value="sequencer",
        label="Select Package to Explore"
    )

    return (package_selector,)


@app.cell(hide_code=True)
def _(
    global_package_paths,
    mo,
    os,
    package_selector,
    print_directory_structure,
):
    # Explore the selected directory
    if not package_selector.value:
        directory_output = mo.md("Please select a package.")
    else:
        output_lines = []

        def explore_single_package(pkg_name):
            if pkg_name not in global_package_paths:
                return [f"❌ Package '{pkg_name}' not configured"]

            pkg_path = global_package_paths[pkg_name]

            if not pkg_path or not os.path.exists(pkg_path):
                return [f"❌ Directory not found: {pkg_path}"]

            lines = [f"## {pkg_name.upper()} Directory Structure"]
            lines.append(f"**Path:** `{pkg_path}`")
            lines.append("")
            lines.append("```")

            structure_lines = print_directory_structure(
                pkg_path,
                exclude_dirs=[
                    '__pycache__', 
                    '.git', 
                    '.pytest_cache',
                    'sps_gui.egg-info',
                    'sequencer.egg-info',
                    'docs',
                    'examples',
                    '.venv',
                    '.venv2',
                    '.venv3',
                    '.venv4', 
                    '.archive', 
                    '.vscode', 
                    '.claude'],
                collapse_dirs=[
                    'icon',
                    'logs', 
                    'data', 
                    'backup', 
                    '_script_testing', 
                    'Auto_Data', 
                    '_del_test',
                    'cli',
                    'core_readmes',
                    'config',
                    '.ALL',
                    'tests',
                    'Tracker_Data']
            )

            lines.extend(structure_lines)
            lines.append("```")
            lines.append("")

            return lines

        if package_selector.value == "both":
            for pkg in ["sequencer", "sps_gui"]:
                output_lines.extend(explore_single_package(pkg))
                output_lines.append("---")
        else:
            output_lines.extend(explore_single_package(package_selector.value))

        directory_output = mo.md("\n".join(output_lines))

    return (directory_output,)


@app.cell(hide_code=True)
def _(directory_output, environment_ui, mo, package_selector, path_status):
    # Main layout
    mo.vstack([
        mo.md("# Package Directory Explorer v10"),
        mo.md("Multi-environment directory structure explorer with enhanced tm_scope individual trace + bulk selection controls + dynamic wafer discovery & push-button filtering."),
        environment_ui,
        path_status,
        mo.md("---"),
        package_selector,
        mo.md("---"),
        directory_output
    ])
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""# Column_B: Duty Cycles to Dataframe Viewer""")
    return


@app.cell
def _():
    ### Adding sequencer.config path
    ### sequencer.config.loaded_duty_dict import loaded_duty_dict
    ### sequencer.config.pyconfig_pickle
    ### sequencer.resources.config.RTK_duty_cycles.pkl
    return


@app.cell(hide_code=True)
def _(global_package_paths, os, resolved_start_path):
    # contains the dataframe processing and pickle functions
    if resolved_start_path and os.path.isdir(resolved_start_path):
        # Use os.path.join for better path construction
        global_package_paths['sequencer.config'] = os.path.join(
            resolved_start_path, "USDCT1_PA", "Parent_Dir", "Data_to_organize", 
            "Lab_Instrument_Doc", "TC2_software", "Efficiency_v2", ".sequencer", 
            "src", "sequencer", "config"
        )

        # contains the dataframe pickle files, sequencer\\resources\\config\\pyconfig_pickle.py
        global_package_paths['sequencer.resources.config'] = os.path.join(
            resolved_start_path, "USDCT1_PA", "Parent_Dir", "Data_to_organize",
            "Lab_Instrument_Doc", "TC2_software", "Efficiency_v2", ".sequencer",
            "src", "sequencer", "resources", "config"
        )

        global_package_paths['sequencer.core.Auto_Data'] = os.path.join(
            resolved_start_path, "USDCT1_PA", "Parent_Dir", "Data_to_organize",
            "Lab_Instrument_Doc", "TC2_software", "Efficiency_v2", ".sequencer",
            "src", "sequencer", "core", "Auto_Data"
        )
    else:
        print(f"resolved_start_path=\n{resolved_start_path} \nis invalid")
    return


@app.cell(hide_code=True)
def _(global_package_paths, mo, os, selected_env):
    try:
        if 'sequencer.config' in global_package_paths and os.path.isdir(global_package_paths['sequencer.config']):
            print("📁 Sequencer paths configured:")
            print(f"  Config: {global_package_paths['sequencer.config']}")
            print(f"  Resources: {global_package_paths['sequencer.resources.config']}")
            print(f"  Auto_Data: {global_package_paths['sequencer.core.Auto_Data']}")
        else:
            validation_message = mo.md(f"""
            ### ⚠️ Path Validation Warning

            **Environment:** {selected_env}  
            **Issue:** Sequencer paths are not accessible or invalid

            Please verify your environment selection and ensure the paths exist.
            """)
            print('Environment paths are invalid. Check environment selection.')
            print(f"Selected environment: {selected_env}")
    except KeyError as e:
        print(f"No key found for global_package_paths, \n{e}")
    return


@app.cell(hide_code=True)
def _(os):
    # Pickle file discovery and UI
    def discover_pickle_files(base_path):
        """Discover all pickle files in the sequencer resources directory"""
        pickle_files = []

        if not base_path:
            return pickle_files

        config_path = os.path.join(base_path, "resources", "config")

        if os.path.exists(config_path):
            for file in os.listdir(config_path):
                if file.endswith('.pkl'):
                    full_path = os.path.join(config_path, file)
                    pickle_files.append({
                        'name': file,
                        'path': full_path,
                        'size': os.path.getsize(full_path) if os.path.exists(full_path) else 0
                    })

        return sorted(pickle_files, key=lambda x: x['name'])

    return (discover_pickle_files,)


@app.cell(hide_code=True)
def _(discover_pickle_files, os, resolved_start_path):
    # Get sequencer base path and discover pickle files
    if resolved_start_path:
        sequencer_base = os.path.join(
            resolved_start_path, "USDCT1_PA", "Parent_Dir", "Data_to_organize",
            "Lab_Instrument_Doc", "TC2_software", "Efficiency_v2", ".sequencer",
            "src", "sequencer"
        )
    else:
        sequencer_base = ""

    pickle_files = discover_pickle_files(sequencer_base)

    # Display discovered files
    if pickle_files:
        print("📁 Discovered Pickle Files:")
        for pf in pickle_files:
            size_kb = pf['size'] / 1024
            print(f"  • {pf['name']} ({size_kb:.1f} KB)")
    else:
        print("❌ No pickle files found")

    return (pickle_files,)


@app.cell(hide_code=True)
def _(mo, pickle_files):
    # Create UI for pickle file selection
    if pickle_files:
        pickle_options = [pf['name'] for pf in pickle_files]
        pickle_selector = mo.ui.dropdown(
            options=pickle_options,
            value=pickle_options[0] if pickle_options else None,
            label="Select Pickle File to View"
        )

        load_button = mo.ui.button(
            label="Load Selected Pickle",
            kind="success",
            value=1,
            on_click=lambda value: value*-1,  # toggle positive/negative with click
        )

        pickle_ui = mo.vstack([
            mo.md("### Pickle File Selection"),
            pickle_selector,
            load_button,
        ])
    else:
        # Create dummy components to maintain Marimo variable consistency
        pickle_selector = mo.ui.dropdown(options=["No files"], value="No files", label="No pickle files available")
        load_button = mo.ui.button(label="No files to load", disabled=True)
        pickle_ui = mo.md("❌ No pickle files available for selection")

    return load_button, pickle_selector, pickle_ui


@app.cell(hide_code=True)
def _(pickle_selector):
    if hasattr(pickle_selector, 'value') and pickle_selector.value:
        print(f"Selected file: {pickle_selector.value}")
    return


@app.cell(hide_code=True)
def _(pickle_ui):
    pickle_ui
    return


@app.cell(hide_code=True)
def _(load_button, pickle_selector):
    print(f"🔍 Debug: Button value = {load_button.value}")
    print(f"🔍 Debug: Selected file = {pickle_selector.value if hasattr(pickle_selector, 'value') else 'None'}")
    print(f"🔍 Debug: Button clicked = {load_button.value is not None}")
    return


@app.cell(hide_code=True)
def _(load_button, pd, pickle, pickle_files, pickle_selector):
    # Get keys organized by layer/depth for dictionary structure
    def private_get_keys_by_layer(d, layer=0, result=None):
        if result is None:
            result = {}

        if isinstance(d, dict):
            if layer not in result:
                result[layer] = set()

            for key, value in d.items():
                result[layer].add(key)
                private_get_keys_by_layer(value, layer + 1, result)

        return result

    # Load selected pickle file when button is clicked (value changes)
    if (load_button.value > 0 and 
        hasattr(pickle_selector, 'value') and 
        pickle_selector.value and 
        pickle_selector.value != "No files"):

        selected_file = next((pf for pf in pickle_files if pf['name'] == pickle_selector.value), None)

        if selected_file:
            try:
                with open(selected_file['path'], 'rb') as f:
                    global_loaded_data = pickle.load(f)

                # Store metadata
                pickle_metadata = {
                    'filename': selected_file['name'],
                    'path': selected_file['path'],
                    'data_type': type(global_loaded_data).__name__,
                    'size_kb': selected_file['size'] / 1024,
                    'load_count': load_button.value  # Track how many times loaded
                }

                print(f"✅ Successfully loaded: {selected_file['name']} (Load #{load_button.value})")
                print(f"📊 Data type: {type(global_loaded_data).__name__}")

                # Try to get more info about the data
                if isinstance(global_loaded_data, dict):
                    keys_by_layer = private_get_keys_by_layer(global_loaded_data)
                    for layer, keys in keys_by_layer.items():
                        print(f"🔑 Layer {layer} keys: {list(keys)}")
                elif isinstance(global_loaded_data, pd.DataFrame):
                    print(f"📈 DataFrame shape: {global_loaded_data.shape}")
                    print(f"📋 Columns: {list(global_loaded_data.columns)}")
                elif isinstance(global_loaded_data, (list, tuple)):
                    print(f"📝 Length: {len(global_loaded_data)}")

            except Exception as e:
                global_loaded_data = None
                pickle_metadata = {'error': str(e)}
                print(f"❌ Error loading {selected_file['name']}: {e}")
        else:
            global_loaded_data = None
            pickle_metadata = {}
    else:
        global_loaded_data = None
        pickle_metadata = {}

    return global_loaded_data, pickle_metadata


@app.cell(hide_code=True)
def _(global_loaded_data, mo, pd, pickle_metadata):
    # Display loaded data information and preview
    if global_loaded_data is not None:
        info_lines = [
            f"## 📁 {pickle_metadata.get('filename', 'Unknown')}",
            f"**Type:** {pickle_metadata.get('data_type', 'Unknown')}",
            f"**Size:** {pickle_metadata.get('size_kb', 0):.1f} KB",
            ""
        ]

        # Create preview based on data type
        if isinstance(global_loaded_data, pd.DataFrame):
            info_lines.extend([
                f"**Shape:** {global_loaded_data.shape[0]} rows × {global_loaded_data.shape[1]} columns",
                f"**Columns:** {', '.join(global_loaded_data.columns)}",
                "",
                "### Data Preview (First 5 rows):"
            ])

            preview_table = mo.ui.table(global_loaded_data.head())
            data_info = mo.vstack([
                mo.md("\n".join(info_lines)),
                preview_table
            ])

        elif isinstance(global_loaded_data, dict):
            info_lines.extend([
                f"**Keys:** {len(global_loaded_data)} items",
                "",
                "### Dictionary Structure:"
            ])

            # Check if this is a duty cycle dictionary
            _is_duty_dict = False
            try:
                _first_key = next(iter(global_loaded_data))
                if isinstance(global_loaded_data[_first_key], dict):
                    _second_level = global_loaded_data[_first_key]
                    _second_key = next(iter(_second_level))
                    if isinstance(_second_level[_second_key], dict):
                        _is_duty_dict = True
            except:
                pass

            if _is_duty_dict:
                # Special handling for duty cycle dictionaries
                dict_preview = ["This appears to be a **Duty Cycle Dictionary** with structure:", ""]
                dict_preview.append("```")
                dict_preview.append("duty_dict[VOUT][PVIN][LOAD] = duty_cycle")
                dict_preview.append("```")
                dict_preview.append("")

                # Show available VOUT values
                _vout_values = sorted(global_loaded_data.keys())
                dict_preview.append(f"**VOUT values:** {', '.join([f'{v}V' for v in _vout_values])}")

                # Show PVIN values for first VOUT
                _first_vout = _vout_values[0]
                _pvin_values = sorted(global_loaded_data[_first_vout].keys())
                dict_preview.append(f"**PVIN values (for VOUT={_first_vout}V):** {', '.join([f'{v}V' for v in _pvin_values])}")

                # Show load values for first VOUT/PVIN combination
                _first_pvin = _pvin_values[0]
                _load_values = sorted(global_loaded_data[_first_vout][_first_pvin].keys())
                dict_preview.append(f"**Load values:** {', '.join([f'{v}A' for v in _load_values])}")

                # Show sample duty cycle
                _sample_duty = global_loaded_data[_first_vout][_first_pvin][_load_values[0]]
                dict_preview.append(f"**Sample duty cycle:** {_sample_duty:.2f}% (at VOUT={_first_vout}V, PVIN={_first_pvin}V, Load={_load_values[0]}A)")

            else:
                # Regular dictionary preview
                dict_preview = []
                for key, value in list(global_loaded_data.items())[:10]:  # Show first 10 items
                    value_type = type(value).__name__
                    if isinstance(value, (list, dict, tuple)):
                        value_summary = f"{value_type} (length: {len(value)})"
                    elif isinstance(value, pd.DataFrame):
                        value_summary = f"DataFrame {value.shape}"
                    else:
                        value_summary = f"{value_type}: {str(value)[:50]}..."
                    dict_preview.append(f"- **{key}**: {value_summary}")

                if len(global_loaded_data) > 10:
                    dict_preview.append(f"... and {len(global_loaded_data) - 10} more items")

            info_lines.extend(dict_preview)
            data_info = mo.md("\n".join(info_lines))

        else:
            info_lines.extend([
                "### Raw Data Preview:",
                f"```\n{str(global_loaded_data)[:500]}{'...' if len(str(global_loaded_data)) > 500 else ''}\n```"
            ])
            data_info = mo.md("\n".join(info_lines))
    else:
        if pickle_metadata and 'error' in pickle_metadata:
            data_info = mo.md(f"❌ **Error:** {pickle_metadata['error']}")
        else:
            data_info = mo.md("Select and load a pickle file to view its contents.")

    return (data_info,)


@app.cell(hide_code=True)
def _(data_info):
    data_info
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""# Column_C: CWD Paths""")
    return


@app.cell
def _(global_package_paths):
    print(global_package_paths.keys())
    return


@app.cell(hide_code=True)
def _(global_package_paths, global_sequencer_path, global_sps_gui_path):
    try:
        print(f"global_sequencer_path:\n {global_sequencer_path}\n")
        print(f"global_sps_gui_path:\n {global_sps_gui_path}\n")

        if 'sequencer.config' in global_package_paths:
            print(f"sequencer.config path:\n {global_package_paths['sequencer.config']}\n")

        print("#--------- Location for duty_cycle dictionary pickle files ---------#")
        if 'sequencer.resources.config' in global_package_paths:
            print(f"sequencer.resources.config path:\n {global_package_paths['sequencer.resources.config']}\n")

        print("#--------- Location for Auto_Data ---------#")
        if 'sequencer.core.Auto_Data' in global_package_paths:
            print(f"sequencer.core.Auto_Data path:\n {global_package_paths['sequencer.core.Auto_Data']}")
    except KeyError as e:
        print(f"No key found.\n{e}")
    return


@app.cell
def _():
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""# Column_D: Generate New Duty Cycle Dictionaries""")
    return


@app.cell
def _():
    # global_package_paths['sequencer.resources.config']
    # TODO 20250528 - need to create a function that creates a new pkl file in above location and prepends marimo_{pkl_filename}
    #               - this should be a copy of TC3-A5-1_test2_update.pkl, but has additional VOUT=0.8V dataseries up to 60A
    # TODO 20250528 - need to create an object that maps CONDITIONS to a dutycycle_vs_load equation
    #               - utilize MPS_df_750-1500khz.html / MPS_df_750-1500khz_duty_cycles.pkl
    return


@app.cell
def _():
    return


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(r"""# Column_E: tm_scope Directory Navigation & File Discovery (v10: Dynamic Wafer Discovery)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Dynamic Wafer Directory Navigation and File Discovery (v10 Enhancement)

    Discover and validate tm_scope Excel files across multiple wafer directories with automatic detection.

    **Enhanced Target Structure (v10):**
    - tm_scope_data/[WAFER_DIR]/[VOUT]/[CBOOT]/Excel_files
    - WAFER directories: Wave1Wafer3_A0-V1, Wave2WaferX_A1-V1, etc. (dynamically discovered)
    - VOUT conditions: 0.8VOUT, 1.8VOUT, etc.
    - CBOOT values: 0.1, 0.22, 0.47 (μF)

    **v10 Enhancements:**
    - Dynamic wafer directory discovery
    - Support for unlimited wafer types
    - Enhanced data hierarchy: WAFER > VOUT > CBOOT > SW/PH > LOAD
    """
    )
    return


@app.cell(hide_code=True)
def _(Any, Dict, List, is_deployment_environment, os, resolved_start_path):
    # Column E: Dynamic Wafer Directory Navigation and File Discovery (v10)
    def discover_tm_scope_files_dynamic(base_path: str) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """
        Discover tm_scope Excel files across multiple wafer directories (v10 Enhancement)

        Returns: {wafer_dir: {vout_dir: {cboot_dir: [files]}}}
        """
        if not base_path or not os.path.exists(base_path):
            return {}

        # v11: Construct tm_scope_data path based on environment
        if is_deployment_environment():
            # Simple path for deployment
            tm_scope_path = os.path.join(base_path, "data", "tm_scope_data")
        else:
            # Original nested path for local development
            tm_scope_path = os.path.join(base_path, 
                "USDCT1_PA", "Parent_Dir", "Data_to_organize", 
                "Lab_Instrument_Doc", "TC2_software", "sps_gui", 
                "src", "sps_gui", "utils", "notebooks", "tm_scope_data")

        if not os.path.exists(tm_scope_path):
            print(f"❌ tm_scope_data directory not found: {tm_scope_path}")
            return {}

        discovered_files = {}

        # Dynamically discover all wafer directories
        try:
            wafer_dirs = [d for d in os.listdir(tm_scope_path) 
                         if os.path.isdir(os.path.join(tm_scope_path, d)) 
                         and ('Wave' in d or 'wafer' in d.lower())]
        except (PermissionError, OSError):
            print(f"❌ Error accessing tm_scope_data directory: {tm_scope_path}")
            return {}

        if not wafer_dirs:
            print(f"❌ No wafer directories found in: {tm_scope_path}")
            return {}

        print(f"🔍 Found {len(wafer_dirs)} wafer directories: {wafer_dirs}")

        for wafer_dir in wafer_dirs:
            wafer_path = os.path.join(tm_scope_path, wafer_dir)
            discovered_files[wafer_dir] = {}

            print(f"  📂 Processing wafer directory: {wafer_dir}")

            # Scan for VOUT directories within each wafer
            try:
                vout_dirs = [d for d in os.listdir(wafer_path) 
                            if os.path.isdir(os.path.join(wafer_path, d)) and 'VOUT' in d]
            except (PermissionError, OSError):
                print(f"    ❌ Error accessing wafer directory: {wafer_path}")
                continue

            for vout_dir in vout_dirs:
                vout_path = os.path.join(wafer_path, vout_dir)
                discovered_files[wafer_dir][vout_dir] = {}

                # Scan for CBOOT subdirectories
                try:
                    cboot_dirs = [d for d in os.listdir(vout_path) 
                                 if os.path.isdir(os.path.join(vout_path, d))]
                except (PermissionError, OSError):
                    continue

                for cboot_dir in cboot_dirs:
                    # Skip .ALL directories during discovery
                    if cboot_dir.startswith('.'):
                        continue

                    cboot_path = os.path.join(vout_path, cboot_dir)
                    excel_files = []
                    try:
                        for file_name in os.listdir(cboot_path):
                            if file_name.endswith('.xlsx') and not file_name.startswith('~'):
                                excel_files.append(file_name)
                    except (PermissionError, OSError) as e:
                        print(f"    ❌ Error accessing {cboot_path}: {e}")
                        continue

                    if excel_files:
                        discovered_files[wafer_dir][vout_dir][cboot_dir] = sorted(excel_files)

                # Check for direct files in VOUT directory
                try:
                    direct_files = [f for f in os.listdir(vout_path) 
                                  if f.endswith('.xlsx') and not f.startswith('~')]
                    if direct_files:
                        discovered_files[wafer_dir][vout_dir]["direct"] = sorted(direct_files)
                except (PermissionError, OSError):
                    pass

        return discovered_files

    def validate_file_structure_with_wafers(discovered_files: Dict) -> Dict[str, Any]:
        """
        Validate file accessibility and extract wafer-level metadata (v10 Enhancement)
        """
        validation_results = {
            'total_files': 0,
            'wafer_directories': [],
            'vout_conditions': [],
            'cboot_values': [],
            'wafer_metadata': {},
            'errors': []
        }

        for wafer_key, wafer_data in discovered_files.items():
            validation_results['wafer_directories'].append(wafer_key)
            validation_results['wafer_metadata'][wafer_key] = {
                'vout_conditions': set(),
                'cboot_values': set(),
                'file_count': 0
            }

            for vout_key, cboot_dict in wafer_data.items():
                # Extract VOUT value from directory name
                try:
                    vout_value = float(vout_key.replace('VOUT', ''))
                    validation_results['vout_conditions'].append(vout_value)
                    validation_results['wafer_metadata'][wafer_key]['vout_conditions'].add(vout_value)
                except ValueError:
                    validation_results['errors'].append(f"Could not parse VOUT from {vout_key} in {wafer_key}")

                for cboot_key, file_list in cboot_dict.items():
                    if cboot_key != "direct":
                        # Extract CBOOT value
                        try:
                            cboot_value = float(cboot_key)
                            validation_results['cboot_values'].append(cboot_value)
                            validation_results['wafer_metadata'][wafer_key]['cboot_values'].add(cboot_value)
                        except ValueError:
                            validation_results['errors'].append(f"Could not parse CBOOT from {cboot_key} in {wafer_key}")

                    file_count = len(file_list)
                    validation_results['total_files'] += file_count
                    validation_results['wafer_metadata'][wafer_key]['file_count'] += file_count

        # Remove duplicates and sort
        validation_results['vout_conditions'] = sorted(list(set(validation_results['vout_conditions'])))
        validation_results['cboot_values'] = sorted(list(set(validation_results['cboot_values'])))

        # Convert sets to lists in wafer metadata
        for wafer_key in validation_results['wafer_metadata']:
            validation_results['wafer_metadata'][wafer_key]['vout_conditions'] = sorted(list(validation_results['wafer_metadata'][wafer_key]['vout_conditions']))
            validation_results['wafer_metadata'][wafer_key]['cboot_values'] = sorted(list(validation_results['wafer_metadata'][wafer_key]['cboot_values']))

        return validation_results

    # Discover files and validate structure with dynamic wafer detection
    tm_scope_files_discovered_v10 = discover_tm_scope_files_dynamic(resolved_start_path)
    tm_file_validation_results_v10 = validate_file_structure_with_wafers(tm_scope_files_discovered_v10)

    # v11: Construct base path for tm_scope data based on environment
    if is_deployment_environment():
        tm_scope_base_path_v10 = os.path.join(resolved_start_path, "data", "tm_scope_data") if resolved_start_path else ""
    else:
        tm_scope_base_path_v10 = os.path.join(resolved_start_path, 
            "USDCT1_PA", "Parent_Dir", "Data_to_organize", 
            "Lab_Instrument_Doc", "TC2_software", "sps_gui", 
            "src", "sps_gui", "utils", "notebooks", "tm_scope_data") if resolved_start_path else ""

    # Display discovery results with wafer breakdown
    if tm_scope_files_discovered_v10:
        print("📁 Discovered tm_scope Excel Files (v10 - Dynamic Wafer Discovery):")
        for wafer_key, wafer_data in tm_scope_files_discovered_v10.items():
            wafer_metadata = tm_file_validation_results_v10['wafer_metadata'].get(wafer_key, {})
            print(f"\n  🧬 {wafer_key} ({wafer_metadata.get('file_count', 0)} files):")
            print(f"    VOUT conditions: {wafer_metadata.get('vout_conditions', [])}")
            print(f"    CBOOT values: {wafer_metadata.get('cboot_values', [])}")

            for vout_key, cboot_dict in wafer_data.items():
                print(f"    📂 {vout_key}:")
                for cboot_key, file_list in cboot_dict.items():
                    print(f"      📁 {cboot_key}: {len(file_list)} files")
                    for file_item in file_list[:2]:  # Show first 2 files
                        print(f"        • {file_item}")
                    if len(file_list) > 2:
                        print(f"        ... and {len(file_list) - 2} more")

        print(f"\n  📊 Overall Validation Results:")
        print(f"    Total wafer directories: {len(tm_file_validation_results_v10['wafer_directories'])}")
        print(f"    Wafer types: {tm_file_validation_results_v10['wafer_directories']}")
        print(f"    Total files: {tm_file_validation_results_v10['total_files']}")
        print(f"    All VOUT conditions: {tm_file_validation_results_v10['vout_conditions']}")
        print(f"    All CBOOT values: {tm_file_validation_results_v10['cboot_values']}")

        if tm_file_validation_results_v10['errors']:
            print(f"    Errors: {tm_file_validation_results_v10['errors']}")
    else:
        print("❌ No tm_scope Excel files found")

    return (
        tm_file_validation_results_v10,
        tm_scope_base_path_v10,
        tm_scope_files_discovered_v10,
    )


@app.cell(hide_code=True)
def _(
    mo,
    tm_file_validation_results_v10,
    tm_scope_base_path_v10,
    tm_scope_files_discovered_v10,
):
    # Create UI for tm_scope file discovery with wafer selection
    if tm_scope_files_discovered_v10:
        # Generate filter options from discovered data including wafer selection
        wafer_options = tm_file_validation_results_v10['wafer_directories'] + ["All"]
        vout_options = [f"{v}V" for v in tm_file_validation_results_v10['vout_conditions']] + ["All"]
        cboot_options = [f"{c}μF" for c in tm_file_validation_results_v10['cboot_values']] + ["All"]

        # Wafer selector (NEW in v10)
        tm_wafer_selector = mo.ui.dropdown(
            options=wafer_options,
            value="All",
            label="Select Wafer Type"
        )

        # Voltage selector
        tm_voltage_selector = mo.ui.dropdown(
            options=vout_options,
            value="All",
            label="Select VOUT Filter"
        )

        # Cboot selector
        tm_cboot_selector = mo.ui.dropdown(
            options=cboot_options,
            value="All",
            label="Select CBOOT Filter"
        )

        # Enhanced file status display with wafer information
        wafer_summary_v10 = []
        for wafer_name_v10 in tm_file_validation_results_v10['wafer_directories']:
            wafer_meta_v10 = tm_file_validation_results_v10['wafer_metadata'][wafer_name_v10]
            wafer_summary_v10.append(f"**{wafer_name_v10}:** {wafer_meta_v10['file_count']} files")

        tm_file_status_display = mo.md(f"""
        ### 📊 File Discovery Status (v10: Dynamic Wafer Discovery)

        **Total Wafer Directories:** {len(tm_file_validation_results_v10['wafer_directories'])}  
        {chr(10).join(wafer_summary_v10)}

        **Total Files Discovered:** {tm_file_validation_results_v10['total_files']}  
        **VOUT Conditions:** {', '.join([f'{v}V' for v in tm_file_validation_results_v10['vout_conditions']])}  
        **CBOOT Values:** {', '.join([f'{c}μF' for c in tm_file_validation_results_v10['cboot_values']])}  
        **Base Path:** `{tm_scope_base_path_v10}`
        """)

        tm_scope_ui = mo.vstack([
            mo.md("### tm_scope File Discovery (v10: Dynamic Wafer Discovery)"),
            tm_file_status_display,
            tm_wafer_selector,
            tm_voltage_selector,
            tm_cboot_selector,
        ])
    else:
        # Create dummy components to maintain Marimo variable consistency
        tm_wafer_selector = mo.ui.dropdown(options=["No data"], value="No data", label="No wafer data available")
        tm_voltage_selector = mo.ui.dropdown(options=["No data"], value="No data", label="No VOUT data available")
        tm_cboot_selector = mo.ui.dropdown(options=["No data"], value="No data", label="No CBOOT data available")
        tm_file_status_display = mo.md("❌ No tm_scope files discovered")
        tm_scope_ui = mo.md("❌ No tm_scope files available for processing")

    return (tm_scope_ui,)


@app.cell(hide_code=True)
def _(tm_scope_ui):
    tm_scope_ui
    return


@app.cell(column=5, hide_code=True)
def _(mo):
    mo.md(r"""# Column_F: Data Extraction & Enhanced TMScopeDataCollection (v10: Wafer Hierarchy)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data Extraction and Enhanced Collection Creation (v10)

    Parse Excel files and build enhanced hierarchical TMScopeDataCollection structure with wafer dimension.

    **Enhanced Data Organization (v10):**
    - WAFER > VOUT > CBOOT > SW/PH > LOAD hierarchy (NEW)
    - Shared time base from CH1-SW_wfm "Time (s)" column  
    - Load mapping from acquisition columns
    - Wafer metadata tracking
    """
    )
    return


@app.cell(hide_code=True)
def _(Any, Dict, List, Optional, dataclass, pd):
    # Column F: Enhanced TMScopeDataCollection with Wafer Hierarchy (v10)
    @dataclass
    class TMScopeDataCollection:
        """
        Enhanced hierarchical data container: WAFER > VOUT > CBOOT > SW/PH > LOAD (v10)

        Structure:
            data[wafer][vout][cboot][channel][load] = pd.Series (y-values)
            time_base[wafer][vout][cboot] = pd.Series (shared x-values from "Time (s)")
            metadata[wafer] = Dict[str, Any] (wafer-specific metadata)
        """
        data: Dict[str, Dict[float, Dict[float, Dict[str, Dict[str, pd.Series]]]]]
        time_base: Dict[str, Dict[float, Dict[float, pd.Series]]]
        metadata: Dict[str, Dict[str, Any]]

        def get_available_wafer_names(self) -> List[str]:
            """Return all available wafer directory names"""
            return sorted(self.data.keys())

        def get_available_vout_values(self, wafer: Optional[str] = None) -> List[float]:
            """Return all available VOUT values, optionally filtered by wafer"""
            if wafer is not None:
                return sorted(self.data.get(wafer, {}).keys())
            else:
                all_vout = set()
                for wafer_data in self.data.values():
                    all_vout.update(wafer_data.keys())
                return sorted(all_vout)

        def get_available_cboot_values(self, wafer: Optional[str] = None, vout: Optional[float] = None) -> List[float]:
            """Return all available CBOOT values, optionally filtered by wafer and/or VOUT"""
            if wafer is not None and vout is not None:
                return sorted(self.data.get(wafer, {}).get(vout, {}).keys())
            elif wafer is not None:
                all_cboot = set()
                for vout_data in self.data.get(wafer, {}).values():
                    all_cboot.update(vout_data.keys())
                return sorted(all_cboot)
            else:
                all_cboot = set()
                for wafer_data in self.data.values():
                    for vout_data in wafer_data.values():
                        all_cboot.update(vout_data.keys())
                return sorted(all_cboot)

        def get_available_channels(self, wafer: Optional[str] = None, vout: Optional[float] = None, cboot: Optional[float] = None) -> List[str]:
            """Return all available channels (SW, PH), optionally filtered"""
            channels = set()

            wafers_to_check = [wafer] if wafer is not None else self.data.keys()
            for w in wafers_to_check:
                vouts_to_check = [vout] if vout is not None else self.data.get(w, {}).keys()
                for v in vouts_to_check:
                    cboots_to_check = [cboot] if cboot is not None else self.data.get(w, {}).get(v, {}).keys()
                    for c in cboots_to_check:
                        channels.update(self.data.get(w, {}).get(v, {}).get(c, {}).keys())
            return sorted(channels)

        def get_available_loads(self, wafer: Optional[str] = None, vout: Optional[float] = None, 
                               cboot: Optional[float] = None, channel: Optional[str] = None) -> List[str]:
            """Return all available load conditions, optionally filtered"""
            loads = set()

            wafers_to_check = [wafer] if wafer is not None else self.data.keys()
            for w in wafers_to_check:
                vouts_to_check = [vout] if vout is not None else self.data.get(w, {}).keys()
                for v in vouts_to_check:
                    cboots_to_check = [cboot] if cboot is not None else self.data.get(w, {}).get(v, {}).keys()
                    for c in cboots_to_check:
                        channels_to_check = [channel] if channel is not None else self.data.get(w, {}).get(v, {}).get(c, {}).keys()
                        for ch in channels_to_check:
                            loads.update(self.data.get(w, {}).get(v, {}).get(c, {}).get(ch, {}).keys())
            return sorted(loads)

    return (TMScopeDataCollection,)


@app.cell(hide_code=True)
def _(Any, Dict, TMScopeDataCollection, is_deployment_environment, os, pd):
    # Enhanced Excel parsing functions for TMScopeDataCollection with wafer hierarchy (v10)
    def parse_excel_to_hierarchical_structure_v10(file_path: str) -> Dict[str, Any]:
        """
        Extract waveform data from single Excel file with wafer detection (v10 Enhancement)
        """
        try:
            # Extract WAFER, VOUT and CBOOT from file path - enhanced parsing for wafer detection
            path_parts = file_path.split(os.sep)
            wafer_dir = None
            vout_dir = None
            cboot_dir = None

            # Find wafer directory (contains 'Wave' or similar pattern)
            for part in path_parts:
                if ('Wave' in part or 'wafer' in part.lower()) and part != file_path.split(os.sep)[-1]:
                    wafer_dir = part
                    break

            # Find VOUT directory (contains 'VOUT' in name)
            for part in path_parts:
                if 'VOUT' in part and part != file_path.split(os.sep)[-1]:  # Not the filename
                    vout_dir = part
                    break

            # Find CBOOT directory (should be numeric like 0.1, 0.22, 0.47)
            for part in path_parts:
                if part != file_path.split(os.sep)[-1]:  # Not the filename
                    # Check if it's a valid cboot value
                    try:
                        cboot_test = float(part)
                        if 0.01 <= cboot_test <= 1.0:  # Reasonable range for cboot in μF
                            cboot_dir = part
                            break
                    except ValueError:
                        continue

            # Skip files in .ALL directories or if we can't extract all values
            if not wafer_dir or not vout_dir or not cboot_dir or '.ALL' in file_path:
                return None

            # Parse WAFER, VOUT and CBOOT values with robust error handling
            try:
                vout_value = float(vout_dir.replace('VOUT', ''))
            except ValueError:
                raise ValueError(f"Could not parse VOUT value from: {vout_dir}")

            try:
                cboot_value = float(cboot_dir)
            except ValueError:
                raise ValueError(f"Could not parse CBOOT value from: {cboot_dir}")

            # Load Excel file with error handling
            try:
                xl_file = pd.ExcelFile(file_path)
            except Exception as e:
                raise ValueError(f"Could not open Excel file: {e}")

            # Extract time base from CH1-SW_wfm sheet
            if 'CH1-SW_wfm' not in xl_file.sheet_names:
                print(f"⚠️ Missing CH1-SW_wfm sheet in {os.path.basename(file_path)}")
                return None

            try:
                ch1_df = pd.read_excel(file_path, sheet_name='CH1-SW_wfm')
            except Exception as e:
                print(f"⚠️ Error reading CH1-SW_wfm sheet: {e}")
                return None

            # Get time data from column A "Time (s)" with fallback options
            time_column = None
            for col in ['Time (s)', 'Time(s)', 'Time', 'time']:
                if col in ch1_df.columns:
                    time_column = col
                    break

            if time_column is None:
                print(f"⚠️ No time column found in {os.path.basename(file_path)}. Available columns: {list(ch1_df.columns)}")
                return None

            time_data = ch1_df[time_column].dropna()
            if len(time_data) == 0:
                print(f"⚠️ Empty time data in {os.path.basename(file_path)}")
                return None

            # Extract CH1 (SW) waveform data
            ch1_data = {}
            acq_columns = [col for col in ch1_df.columns if col.startswith('Acq_') and 'CH1' in col]

            for col in acq_columns:
                try:
                    # Extract acquisition number more robustly
                    parts = col.split('_')
                    if len(parts) >= 2:
                        acq_num = int(parts[1])
                        load_current = map_acquisition_to_load(acq_num)
                        load_name = f"Load_{load_current}A"

                        # Only store non-empty data
                        col_data = ch1_df[col].dropna()
                        if len(col_data) > 0:
                            ch1_data[load_name] = col_data
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Could not parse acquisition column {col}: {e}")
                    continue

            # Extract CH3 (PH) waveform data if available
            ch3_data = {}
            if 'CH3-PH_wfm' in xl_file.sheet_names:
                try:
                    ch3_df = pd.read_excel(file_path, sheet_name='CH3-PH_wfm')
                    acq_columns_ch3 = [col for col in ch3_df.columns if col.startswith('Acq_') and 'CH3' in col]

                    for col in acq_columns_ch3:
                        try:
                            parts = col.split('_')
                            if len(parts) >= 2:
                                acq_num = int(parts[1])
                                load_current = map_acquisition_to_load(acq_num)
                                load_name = f"Load_{load_current}A"

                                # Only store non-empty data
                                col_data = ch3_df[col].dropna()
                                if len(col_data) > 0:
                                    ch3_data[load_name] = col_data
                        except (ValueError, IndexError) as e:
                            print(f"⚠️ Could not parse CH3 acquisition column {col}: {e}")
                            continue
                except Exception as e:
                    print(f"⚠️ Error reading CH3-PH_wfm sheet: {e}")

            # Ensure we have some data to work with
            if not ch1_data and not ch3_data:
                print(f"⚠️ No valid acquisition data found in {os.path.basename(file_path)}")
                return None

            return {
                "wafer": wafer_dir,  # NEW in v10
                "vout": vout_value,
                "cboot": cboot_value,
                "time_data": time_data,
                "ch1_data": ch1_data,
                "ch3_data": ch3_data,
                "load_mapping": get_load_mapping(),
                "metadata": {
                    "filename": os.path.basename(file_path),
                    "file_path": file_path,
                    "wafer_directory": wafer_dir,  # NEW in v10
                    "sheets_found": xl_file.sheet_names
                }
            }

        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")
            return None

    def map_acquisition_to_load(acq_num: int) -> int:
        """Map acquisition number to load current value"""
        # Based on provided mapping from plan
        load_mapping = {
            1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
            11: 9, 12: 10, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20,
            19: 25, 20: 28, 21: 29, 22: 30, 23: 31, 24: 35, 25: 40, 26: 45,
            27: 50, 28: 55, 29: 60, 30: 65, 31: 70
        }
        return load_mapping.get(acq_num, acq_num)  # Default to acq_num if not in mapping

    def get_load_mapping() -> Dict[int, int]:
        """Return the acquisition to load mapping dictionary"""
        return {
            1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
            11: 9, 12: 10, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20,
            19: 25, 20: 28, 21: 29, 22: 30, 23: 31, 24: 35, 25: 40, 26: 45,
            27: 50, 28: 55, 29: 60, 30: 65, 31: 70
        }

    def build_tmscope_collection_v10(discovered_files: Dict, base_path: str) -> TMScopeDataCollection:
        """
        Enhanced collection builder from discovered files with wafer hierarchy (v10)
        """
        collection_data = {}
        collection_time_base = {}
        collection_metadata = {}
        extraction_metadata = {
            "files_processed": 0,
            "files_failed": 0,
            "wafers_found": set(),
            "conditions_found": set(),
            "processing_errors": []
        }

        for wafer_dir, wafer_data in discovered_files.items():
            extraction_metadata["wafers_found"].add(wafer_dir)

            for vout_dir, cboot_dict in wafer_data.items():
                for cboot_dir, file_list in cboot_dict.items():
                    if cboot_dir == "direct":
                        continue  # Skip direct files for now

                    # Process first file from each condition
                    if file_list:
                        # v11: Construct file path based on environment
                        if is_deployment_environment():
                            file_path = os.path.join(base_path, "data", "tm_scope_data", 
                                                   wafer_dir, vout_dir, cboot_dir, file_list[0])
                        else:
                            file_path = os.path.join(base_path, wafer_dir, vout_dir, cboot_dir, file_list[0])

                        # Only process files that are not in skip directories
                        if '.ALL' in file_path:
                            print(f"⏭️ Skipping .ALL directory file: {os.path.basename(file_path)}")
                            continue

                        parsed_data = parse_excel_to_hierarchical_structure_v10(file_path)
                        if parsed_data is not None:
                            wafer_from_parsed = parsed_data["wafer"]
                            vout_from_parsed = parsed_data["vout"]
                            cboot_from_parsed = parsed_data["cboot"]

                            # Initialize hierarchical structure with wafer dimension
                            if wafer_from_parsed not in collection_data:
                                collection_data[wafer_from_parsed] = {}
                                collection_time_base[wafer_from_parsed] = {}
                                collection_metadata[wafer_from_parsed] = {
                                    "wafer_directory": wafer_from_parsed,
                                    "conditions": set(),
                                    "files_processed": 0
                                }

                            if vout_from_parsed not in collection_data[wafer_from_parsed]:
                                collection_data[wafer_from_parsed][vout_from_parsed] = {}
                                collection_time_base[wafer_from_parsed][vout_from_parsed] = {}

                            if cboot_from_parsed not in collection_data[wafer_from_parsed][vout_from_parsed]:
                                collection_data[wafer_from_parsed][vout_from_parsed][cboot_from_parsed] = {}
                                collection_time_base[wafer_from_parsed][vout_from_parsed][cboot_from_parsed] = parsed_data["time_data"]

                            # Store channel data
                            if parsed_data["ch1_data"]:
                                collection_data[wafer_from_parsed][vout_from_parsed][cboot_from_parsed]["SW"] = parsed_data["ch1_data"]

                            if parsed_data["ch3_data"]:
                                collection_data[wafer_from_parsed][vout_from_parsed][cboot_from_parsed]["PH"] = parsed_data["ch3_data"]

                            # Update metadata
                            extraction_metadata["files_processed"] += 1
                            collection_metadata[wafer_from_parsed]["files_processed"] += 1
                            condition_id = f"{wafer_from_parsed}_VOUT_{vout_from_parsed}V_CBOOT_{cboot_from_parsed}μF"
                            extraction_metadata["conditions_found"].add(condition_id)
                            collection_metadata[wafer_from_parsed]["conditions"].add(condition_id)

                            print(f"✅ Processed: {condition_id} from {os.path.basename(file_path)}")
                        else:
                            extraction_metadata["files_failed"] += 1
                            print(f"⏭️ Skipped: {os.path.basename(file_path)} (unsupported format or .ALL directory)")

        # Convert sets to lists for JSON serialization
        extraction_metadata["wafers_found"] = list(extraction_metadata["wafers_found"])
        extraction_metadata["conditions_found"] = list(extraction_metadata["conditions_found"])

        for wafer_in_metadata in collection_metadata:
            if wafer_in_metadata != "_extraction_summary":
                collection_metadata[wafer_in_metadata]["conditions"] = list(collection_metadata[wafer_in_metadata]["conditions"])

        # Add extraction metadata to collection metadata
        collection_metadata["_extraction_summary"] = extraction_metadata

        return TMScopeDataCollection(
            data=collection_data,
            time_base=collection_time_base,
            metadata=collection_metadata
        )

    return (build_tmscope_collection_v10,)


@app.cell(hide_code=True)
def _(
    TMScopeDataCollection,
    build_tmscope_collection_v10,
    tm_scope_base_path_v10,
    tm_scope_files_discovered_v10,
):
    # Build enhanced TMScopeDataCollection from discovered files with wafer hierarchy (v10)
    if tm_scope_files_discovered_v10 and tm_scope_base_path_v10:
        print("🔄 Building Enhanced TMScopeDataCollection with wafer hierarchy (v10)...")

        tm_scope_collection_v10 = build_tmscope_collection_v10(
            tm_scope_files_discovered_v10,
            tm_scope_base_path_v10
        )

        # Display extraction results with wafer breakdown
        if tm_scope_collection_v10.data:
            extraction_metadata = tm_scope_collection_v10.metadata.get("_extraction_summary", {})
            print(f"✅ Enhanced TMScopeDataCollection (v10) built successfully!")
            print(f"   Files processed: {extraction_metadata.get('files_processed', 0)}")
            print(f"   Files failed: {extraction_metadata.get('files_failed', 0)}")
            print(f"   Wafers found: {len(extraction_metadata.get('wafers_found', []))}")
            print(f"   Wafer types: {extraction_metadata.get('wafers_found', [])}")
            print(f"   Total conditions: {len(extraction_metadata.get('conditions_found', []))}")

            # Show available data structure with wafer dimension
            print(f"\n📊 Available Data (v10 - Wafer Hierarchy):")
            for wafer_name_in_collection in tm_scope_collection_v10.get_available_wafer_names():
                wafer_meta_in_collection = tm_scope_collection_v10.metadata.get(wafer_name_in_collection, {})
                print(f"   🧬 {wafer_name_in_collection} ({wafer_meta_in_collection.get('files_processed', 0)} files):")

                for vout_in_collection in tm_scope_collection_v10.get_available_vout_values(wafer_name_in_collection):
                    print(f"     VOUT {vout_in_collection}V:")
                    for cboot_in_collection in tm_scope_collection_v10.get_available_cboot_values(wafer_name_in_collection, vout_in_collection):
                        channels_in_collection = tm_scope_collection_v10.get_available_channels(wafer_name_in_collection, vout_in_collection, cboot_in_collection)
                        loads_in_collection = tm_scope_collection_v10.get_available_loads(wafer_name_in_collection, vout_in_collection, cboot_in_collection)
                        print(f"       CBOOT {cboot_in_collection}μF: {len(channels_in_collection)} channels, {len(loads_in_collection)} loads")

            data_quality_report_v10 = {
                "total_wafers": len(extraction_metadata.get("wafers_found", [])),
                "wafer_names": extraction_metadata.get("wafers_found", []),
                "total_conditions": len(extraction_metadata.get("conditions_found", [])),
                "vout_values": tm_scope_collection_v10.get_available_vout_values(),
                "cboot_values": tm_scope_collection_v10.get_available_cboot_values(),
                "extraction_success_rate": (
                    extraction_metadata.get("files_processed", 0) / 
                    (extraction_metadata.get("files_processed", 0) + extraction_metadata.get("files_failed", 0))
                    if extraction_metadata.get("files_processed", 0) + extraction_metadata.get("files_failed", 0) > 0 else 0
                )
            }
        else:
            print("❌ Failed to build Enhanced TMScopeDataCollection - no data extracted")
            tm_scope_collection_v10 = TMScopeDataCollection(data={}, time_base={}, metadata={})
            data_quality_report_v10 = {"error": "No data extracted"}
    else:
        print("⚠️ No files discovered or base path not available")
        tm_scope_collection_v10 = TMScopeDataCollection(data={}, time_base={}, metadata={})
        data_quality_report_v10 = {"error": "No files discovered"}

    return data_quality_report_v10, tm_scope_collection_v10


@app.cell(hide_code=True)
def _(data_quality_report_v10, mo, tm_scope_collection_v10):
    # Display enhanced collection summary with wafer information
    if tm_scope_collection_v10.data:
        collection_summary_v10 = mo.md(f"""
        ### 📊 Enhanced TMScopeDataCollection Summary (v10: Wafer Hierarchy)

        **Total Wafers:** {data_quality_report_v10.get('total_wafers', 0)}  
        **Wafer Types:** {', '.join(data_quality_report_v10.get('wafer_names', []))}  
        **Total Conditions:** {data_quality_report_v10.get('total_conditions', 0)}  
        **VOUT Values:** {', '.join([f'{v}V' for v in data_quality_report_v10.get('vout_values', [])])}  
        **CBOOT Values:** {', '.join([f'{c}μF' for c in data_quality_report_v10.get('cboot_values', [])])}  
        **Extraction Success Rate:** {data_quality_report_v10.get('extraction_success_rate', 0):.1%}

        **Enhanced Data Structure (v10):** WAFER > VOUT > CBOOT > Channel (SW/PH) > Load  
        **Time Base:** Shared from CH1-SW_wfm "Time (s)" column per wafer/VOUT/CBOOT
        """)

        data_validation_status_v10 = mo.md("✅ Enhanced data extraction (v10) completed successfully")
    else:
        collection_summary_v10 = mo.md("❌ No data available in Enhanced TMScopeDataCollection")
        data_validation_status_v10 = mo.md(f"❌ Enhanced data extraction failed: {data_quality_report_v10.get('error', 'Unknown error')}")

    return collection_summary_v10, data_validation_status_v10


@app.cell(hide_code=True)
def _(collection_summary_v10, data_validation_status_v10, mo):
    mo.vstack([
        mo.md("### Enhanced Data Extraction Results (v10)"),
        collection_summary_v10,
        data_validation_status_v10
    ])
    return


@app.cell(column=6, hide_code=True)
def _(mo):
    mo.md(r"""# Column_G: Enhanced Plotly Visualization with Wafer Selection + Push-Button Filtering (v10)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Enhanced Interactive Plotly Visualization with Wafer Selection + Push-Button Filtering (v10)

    Create comprehensive waveform plot with enhanced wafer support:
    - **v10 Enhancement**: Dynamic wafer directory discovery and selection
    - **Wafer Checkbox Control**: Select specific wafer types (Wave1Wafer3_A0-V1, Wave2WaferX_A1-V1, etc.)
    - **Plotly-resampler**: Automatic aggregation for large datasets (7.5M+ points)
    - **Individual trace control**: Each load condition independently selectable
    - **Bulk selection controls**: WAFER, VOUT, CBOOT, Channel level filtering
    - **Push-button filtering**: Dynamic buttons for WAFER, CH, VOUT, CBOOT, LOAD parameters
    - **Intensity-based coloring**: Workbook base color + load intensity
    - **Zoom-to-detail**: Full resolution data when zoomed in
    - **Smart defaults**: Representative traces shown for large datasets

    **v10 Data Hierarchy**: WAFER > VOUT > CBOOT > SW/PH > LOAD
    **Enhanced Trace Naming**: Includes wafer identifier for disambiguation
    **Memory Optimization**: 449MB → <10MB output via automatic data aggregation
    """
    )
    return


@app.cell(hide_code=True)
def _(TMScopeDataCollection):
    # Enhanced parameter extraction from tm_scope_collection with wafer support (v10)
    def extract_parameter_values_v10(collection: TMScopeDataCollection) -> tuple:
        """Extract all unique parameter values including wafer names from the collection for button creation (v10)"""
        if not collection.data:
            return [], [], [], [], []

        wafer_names = set()
        channels = set()
        vout_values = set()
        cboot_values = set()
        load_values = set()

        for wafer_for_params in collection.get_available_wafer_names():
            wafer_names.add(wafer_for_params)
            for vout_for_params in collection.get_available_vout_values(wafer_for_params):
                vout_values.add(vout_for_params)
                for cboot_for_params in collection.get_available_cboot_values(wafer_for_params, vout_for_params):
                    cboot_values.add(cboot_for_params)
                    for channel_for_params in collection.get_available_channels(wafer_for_params, vout_for_params, cboot_for_params):
                        channels.add(channel_for_params)
                        for load_for_params in collection.get_available_loads(wafer_for_params, vout_for_params, cboot_for_params, channel_for_params):
                            # Extract numeric load value
                            try:
                                load_numeric_for_params = float(load_for_params.replace("Load_", "").replace("A", ""))
                                load_values.add(load_numeric_for_params)
                            except ValueError:
                                continue

        return (
            sorted(wafer_names),
            sorted(channels),
            sorted(vout_values),
            sorted(cboot_values),
            sorted(load_values)
        )

    return (extract_parameter_values_v10,)


@app.cell(hide_code=True)
def _(extract_parameter_values_v10, tm_scope_collection_v10):
    # Extract parameter values for dynamic button creation including wafer names (v10)
    g_wafer_values, g_ch_values, g_vout_values, g_cboot_values, g_load_values = extract_parameter_values_v10(tm_scope_collection_v10)

    print("📋 Enhanced Parameter Extraction Results (v10):")
    print(f"   Wafer names: {g_wafer_values}")  # NEW in v10
    print(f"   Channels: {g_ch_values}")
    print(f"   VOUT values: {g_vout_values}")
    print(f"   CBOOT values: {g_cboot_values}")
    print(f"   Load values: {g_load_values}")

    return (
        g_cboot_values,
        g_ch_values,
        g_load_values,
        g_vout_values,
        g_wafer_values,
    )


@app.cell(hide_code=True)
def _(
    g_cboot_values,
    g_ch_values,
    g_load_values,
    g_vout_values,
    g_wafer_values,
    mo,
):
    # Create enhanced checkbox UI components with wafer selection (v10)
    def create_checkbox_groups_v10(wafer_vals, ch_vals, vout_vals, cboot_vals, load_vals):
        """Create checkbox groups for each parameter category including wafer selection (v10)"""

        # Wafer checkboxes (Wave1Wafer3_A0-V1, Wave2WaferX_A1-V1, etc.) - NEW in v10
        if wafer_vals:
            g_wafer_checkboxes = mo.ui.array([
                mo.ui.checkbox(label=wafer, value=False)
                for wafer in wafer_vals
            ])
        else:
            g_wafer_checkboxes = mo.ui.array([mo.ui.checkbox(label="No wafers", disabled=True)])

        # Channel checkboxes (SW, PH)
        if ch_vals:
            g_ch_checkboxes = mo.ui.array([
                mo.ui.checkbox(label=ch, value=False)
                for ch in ch_vals
            ])
        else:
            g_ch_checkboxes = mo.ui.array([mo.ui.checkbox(label="No channels", disabled=True)])

        # VOUT checkboxes (0.8V, 1.8V, etc.)
        if vout_vals:
            g_vout_checkboxes = mo.ui.array([
                mo.ui.checkbox(label=f"{v}V", value=False)
                for v in vout_vals
            ])
        else:
            g_vout_checkboxes = mo.ui.array([mo.ui.checkbox(label="No VOUT", disabled=True)])

        # CBOOT checkboxes (0.1μF, 0.22μF, 0.47μF, etc.)
        if cboot_vals:
            g_cboot_checkboxes = mo.ui.array([
                mo.ui.checkbox(label=f"{c}μF", value=False)
                for c in cboot_vals
            ])
        else:
            g_cboot_checkboxes = mo.ui.array([mo.ui.checkbox(label="No CBOOT", disabled=True)])

        # Load checkboxes (0A, 1A, 2A, ..., 70A)
        if load_vals:
            g_load_checkboxes = mo.ui.array([
                mo.ui.checkbox(label=f"{int(load)}A", value=False)
                for load in load_vals
            ])
        else:
            g_load_checkboxes = mo.ui.array([mo.ui.checkbox(label="No loads", disabled=True)])

        return g_wafer_checkboxes, g_ch_checkboxes, g_vout_checkboxes, g_cboot_checkboxes, g_load_checkboxes

    # Create enhanced checkbox groups with extracted values including wafer selection
    g_wafer_checkboxes, g_ch_checkboxes, g_vout_checkboxes, g_cboot_checkboxes, g_load_checkboxes = create_checkbox_groups_v10(
        g_wafer_values, g_ch_values, g_vout_values, g_cboot_values, g_load_values
    )

    def create_load_rows(load_checkboxes, max_per_row=10):
        """Break load checkboxes into multiple rows"""
        if len(load_checkboxes) <= max_per_row:
            return mo.hstack(load_checkboxes, gap=1)
    
        rows = []
        for i in range(0, len(load_checkboxes), max_per_row):
            row_checkboxes = load_checkboxes[i:i + max_per_row]
            rows.append(mo.hstack(row_checkboxes, gap=1))
    
        return mo.vstack(rows, gap=1)

    # Create enhanced organized checkbox panel layout with wafer row
    g_checkbox_panel_v10 = mo.vstack([
        mo.md("### 🎛️ Trace Selection Controls"),
        mo.md("*Check parameters to filter displayed traces. Checked items are selected. Multiple selections are additive.*"),
        mo.md("---"),
        mo.vstack([
            mo.vstack([mo.md("**Wafer:**"), mo.hstack(g_wafer_checkboxes, gap=2)], gap=1),  # NEW in v10
            mo.vstack([mo.md("**Channel:**"), mo.hstack(g_ch_checkboxes, gap=2)], gap=1),
            mo.vstack([mo.md("**VOUT:**"), mo.hstack(g_vout_checkboxes, gap=2)], gap=1),
            mo.vstack([mo.md("**CBOOT:**"), mo.hstack(g_cboot_checkboxes, gap=2)], gap=1),
            #mo.vstack([mo.md("**LOAD:**"), mo.hstack(g_load_checkboxes, gap=2)], gap=1),
            mo.vstack([mo.md("**LOAD:**"), create_load_rows(g_load_checkboxes, 13)], gap=1)
        ], gap=3)
    ])

    return (
        g_cboot_checkboxes,
        g_ch_checkboxes,
        g_checkbox_panel_v10,
        g_load_checkboxes,
        g_vout_checkboxes,
        g_wafer_checkboxes,
    )


@app.cell(hide_code=True)
def _(
    g_cboot_checkboxes,
    g_cboot_values,
    g_ch_checkboxes,
    g_ch_values,
    g_load_checkboxes,
    g_load_values,
    g_vout_checkboxes,
    g_vout_values,
    g_wafer_checkboxes,
    g_wafer_values,
):
    # Process enhanced checkbox states including wafer selection (v10)
    def extract_selected_values_from_checkboxes_v10(checkbox_array, value_list):
        """Extract selected values from checkbox array states - fixed UIElement evaluation (v10)"""
        # Check if arrays exist and have content (avoid UIElement boolean evaluation)
        if checkbox_array is None or value_list is None:
            return []
        if len(checkbox_array) == 0 or len(value_list) == 0:
            return []

        selected = []
        for i, checkbox in enumerate(checkbox_array):
            # Only check .value property - avoid UIElement boolean evaluation
            if i < len(value_list) and checkbox.value:
                selected.append(value_list[i])
        return selected

    # Extract current selections from checkboxes including wafer selection
    g_selected_wafers = extract_selected_values_from_checkboxes_v10(g_wafer_checkboxes, g_wafer_values)  # NEW in v10
    g_selected_channels = extract_selected_values_from_checkboxes_v10(g_ch_checkboxes, g_ch_values)
    g_selected_vout = extract_selected_values_from_checkboxes_v10(g_vout_checkboxes, g_vout_values)
    g_selected_cboot = extract_selected_values_from_checkboxes_v10(g_cboot_checkboxes, g_cboot_values)
    g_selected_loads = extract_selected_values_from_checkboxes_v10(g_load_checkboxes, g_load_values)

    # Create enhanced selection state dictionary with wafer selection
    g_selection_state_v10 = {
        "wafers": g_selected_wafers,     # NEW in v10
        "channels": g_selected_channels,
        "vout": g_selected_vout,
        "cboot": g_selected_cboot,
        "loads": g_selected_loads,
        "any_selected": len(g_selected_wafers) > 0 or len(g_selected_channels) > 0 or len(g_selected_vout) > 0 or len(g_selected_cboot) > 0 or len(g_selected_loads) > 0
    }

    # Display current selections for debugging
    if g_selection_state_v10["any_selected"]:
        print("🎯 Current Enhanced Checkbox Selections (v10):")
        if g_selected_wafers:
            print(f"   Wafers: {g_selected_wafers}")  # NEW in v10
        if g_selected_channels:
            print(f"   Channels: {g_selected_channels}")
        if g_selected_vout:
            print(f"   VOUT: {g_selected_vout}")
        if g_selected_cboot:
            print(f"   CBOOT: {g_selected_cboot}")
        if g_selected_loads:
            print(f"   Load: {g_selected_loads}")

    return (g_selection_state_v10,)


@app.cell(hide_code=True)
def _(Dict, FigureWidgetResampler, TMScopeDataCollection, go, pc):
    # Enhanced Plotly Visualization Functions with Wafer Support + Push-Button Filtering (v10)

    def create_detailed_trace_name_v10(wafer: str, vout: float, cboot: float, channel: str, load_name: str) -> str:
        """Create clear, descriptive trace names including wafer information (v10)"""
        # Extract numeric load value for sorting
        try:
            load_numeric = float(load_name.replace("Load_", "").replace("A", ""))
        except ValueError:
            load_numeric = 0

        # Create hierarchical name with wafer prefix (v10 enhancement)
        return f"{wafer}_{vout}V_{cboot}μF_{channel}_{load_numeric:g}A"

    def should_include_trace_v10(wafer: str, vout: float, cboot: float, channel: str, load_name: str, selection_filters: Dict) -> bool:
        """Determine if trace should be included based on button selections including wafer (v10)"""
        if not selection_filters or not selection_filters.get("any_selected", False):
            return True  # Show all traces if no filters selected

        # Extract numeric load for comparison
        try:
            load_numeric = float(load_name.replace("Load_", "").replace("A", ""))
        except ValueError:
            return False

        # Check each filter category - if category has selections, trace must match at least one
        filters_to_check = []

        if selection_filters.get("wafers"):          # NEW in v10
            filters_to_check.append(wafer in selection_filters["wafers"])

        if selection_filters.get("channels"):
            filters_to_check.append(channel in selection_filters["channels"])

        if selection_filters.get("vout"):
            filters_to_check.append(vout in selection_filters["vout"])

        if selection_filters.get("cboot"):
            filters_to_check.append(cboot in selection_filters["cboot"])

        if selection_filters.get("loads"):
            filters_to_check.append(load_numeric in selection_filters["loads"])

        # If any filter categories are selected, trace must match ALL selected categories
        return all(filters_to_check) if filters_to_check else True

    def configure_individual_trace_legend_v10(fig: FigureWidgetResampler) -> FigureWidgetResampler:
        """Configure legend for individual trace control with wafer support (v10)"""

        fig.update_layout(
            legend=dict(
                title=dict(
                    text="Individual Traces (v10)<br><i>Click to show/hide each trace<br>Format: WAFER_VOUT_CBOOT_CH_LOAD</i>",
                    font=dict(size=12, color="black")
                ),
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1,
                font=dict(size=9),  # Smaller font for many individual entries
                itemsizing="constant",
                itemwidth=30,
                # Allow scrolling for large numbers of traces
                itemclick="toggle",  # Standard individual toggle behavior
                itemdoubleclick="toggleothers"  # Double-click to isolate trace
            )
        )

        return fig

    def apply_smart_default_visibility_v10(fig: FigureWidgetResampler, 
                                         collection: TMScopeDataCollection,
                                         selection_filters: Dict = None) -> FigureWidgetResampler:
        """Apply intelligent default visibility for datasets with many traces including wafer filtering (v10)"""

        total_traces = len(fig.data)

        # If button filters are active, show all matching traces
        if selection_filters and selection_filters.get("any_selected", False):
            print(f"🎯 Enhanced button filters active (v10) - showing {sum(1 for trace in fig.data if trace.visible)} filtered traces")
            return fig

        if total_traces > 30:
            # Show only representative load conditions by default
            representative_loads = ['0A', '10A', '20A', '40A', '60A']

            for trace in fig.data:
                # Check if this trace represents a representative load
                trace_is_representative = any(f"_{load}" in trace.name for load in representative_loads)
                trace.visible = trace_is_representative

            print(f"📊 Large dataset detected ({total_traces} traces)")
            print(f"🎯 Showing representative loads by default: {representative_loads}")
            print("💡 Use enhanced button controls, bulk controls or legend to show additional traces")

        return fig

    def create_enhanced_resampled_plot_v10(collection: TMScopeDataCollection, selection_filters: Dict = None) -> FigureWidgetResampler:
        """
        Enhanced resampled waveform plot with wafer support + push-button filtering (v10)

        v10 Features:
        - Wafer selection: Filter traces by wafer type (Wave1Wafer3_A0-V1, Wave2WaferX_A1-V1, etc.)
        - Enhanced push-button filtering: Filter traces by WAFER, CH, VOUT, CBOOT, LOAD
        - Individual trace legend control (each load condition selectable)
        - Bulk selection via updatemenus (WAFER, VOUT, CBOOT, Channel)
        - Smart defaults for large datasets
        - Enhanced trace naming with wafer identifier

        Features:
        - Single plot shows all data from collection across all wafers
        - Individual trace control: Each load condition independently selectable
        - Bulk selection controls: WAFER, VOUT, CBOOT, Channel level filtering
        - Intensity-based coloring: workbook color + load intensity
        - Zoom-to-detail: Full resolution data when zoomed in
        - Smart defaults: Representative traces shown for large datasets
        """

        if not collection.data:
            # Return empty plot if no data
            fig = FigureWidgetResampler(go.Figure())
            fig.update_layout(
                title="No tm_scope data available",
                xaxis_title="Time (s)", 
                yaxis_title="Voltage (V)",
                height=600
            )
            return fig

        # Create resampler-enabled figure
        fig = FigureWidgetResampler(go.Figure())

        # Performance tracking
        total_traces = 0
        filtered_traces = 0

        # Base colors for different workbooks (WAFER/VOUT/CBOOT combinations)
        base_colors = ["blue", "red", "green", "orange", "purple", "brown",
                      "pink", "gray", "olive", "cyan", "magenta", "yellow"]

        # Get all available loads and normalize for intensity mapping
        all_loads_numeric = []
        for wafer_for_loads in collection.get_available_wafer_names():
            for vout_for_loads in collection.get_available_vout_values(wafer_for_loads):
                for cboot_for_loads in collection.get_available_cboot_values(wafer_for_loads, vout_for_loads):
                    for channel_for_loads in collection.get_available_channels(wafer_for_loads, vout_for_loads, cboot_for_loads):
                        for load_name_for_loads in collection.get_available_loads(wafer_for_loads, vout_for_loads, cboot_for_loads, channel_for_loads):
                            try:
                                load_numeric_for_loads = float(load_name_for_loads.replace("Load_", "").replace("A", ""))
                                all_loads_numeric.append(load_numeric_for_loads)
                            except ValueError:
                                continue

        min_load = min(all_loads_numeric) if all_loads_numeric else 0
        max_load = max(all_loads_numeric) if all_loads_numeric else 100

        # Add traces to the plot (with enhanced filtering including wafer)
        workbook_index = 0
        for wafer_for_plot in collection.get_available_wafer_names():
            for vout_for_plot in collection.get_available_vout_values(wafer_for_plot):
                for cboot_for_plot in collection.get_available_cboot_values(wafer_for_plot, vout_for_plot):
                    # Assign base color for this workbook
                    base_color = base_colors[workbook_index % len(base_colors)]
                    workbook_index += 1

                    # Get shared time base for this workbook
                    time_data = collection.time_base[wafer_for_plot][vout_for_plot][cboot_for_plot]

                    for channel_for_plot in collection.get_available_channels(wafer_for_plot, vout_for_plot, cboot_for_plot):
                        for load_name_for_plot, voltage_series in collection.data[wafer_for_plot][vout_for_plot][cboot_for_plot][channel_for_plot].items():

                            total_traces += 1

                            # Apply enhanced push-button filtering including wafer
                            if not should_include_trace_v10(wafer_for_plot, vout_for_plot, cboot_for_plot, channel_for_plot, load_name_for_plot, selection_filters):
                                continue  # Skip this trace based on button filters

                            filtered_traces += 1

                            # Calculate color intensity based on load
                            try:
                                load_numeric_for_plot = float(load_name_for_plot.replace("Load_", "").replace("A", ""))
                                if max_load > min_load:
                                    intensity = 0.3 + 0.7 * (load_numeric_for_plot - min_load) / (max_load - min_load)
                                else:
                                    intensity = 1.0
                            except ValueError:
                                intensity = 1.0

                            # Create intensity-adjusted color
                            if hasattr(pc.qualitative, 'Plotly'):
                                color_palette = pc.qualitative.Plotly
                            else:
                                color_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                                               '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

                            base_color_hex = color_palette[base_colors.index(base_color) % len(color_palette)]

                            # Convert hex to RGB for intensity adjustment
                            hex_color = base_color_hex.lstrip('#')
                            rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                            adjusted_color = f"rgba({rgb_color[0]},{rgb_color[1]},{rgb_color[2]},{intensity})"

                            # Create enhanced trace name including wafer information (v10)
                            trace_name = create_detailed_trace_name_v10(wafer_for_plot, vout_for_plot, cboot_for_plot, channel_for_plot, load_name_for_plot)

                            # Add trace with individual control (NO legendgroup)
                            fig.add_trace(
                                go.Scattergl(
                                    mode='lines',
                                    name=trace_name,  # Individual legend entry with wafer info
                                    line=dict(color=adjusted_color, width=1.5),
                                    # NO legendgroup = individual control
                                    hovertemplate=(
                                        f'<b>Wafer: {wafer_for_plot}</b><br>'  # NEW in v10
                                        f'<b>VOUT: {vout_for_plot}V, CBOOT: {cboot_for_plot}μF</b><br>'
                                        f'Channel: {channel_for_plot}<br>'
                                        f'Load: {load_name_for_plot}<br>'
                                        'Time: %{x}s<br>'
                                        'Voltage: %{y}V<extra></extra>'
                                    ),
                                    visible=True  # Will be adjusted by smart defaults
                                ),
                                hf_x=time_data,      # High-frequency x data
                                hf_y=voltage_series  # High-frequency y data - auto-resampled!
                            )

        # Configure layout with enhanced individual legend (bulk controls replaced by checkboxes)
        filter_info = ""
        if selection_filters and selection_filters.get("any_selected", False):
            filter_info = f" | {filtered_traces}/{total_traces} traces"

        fig.update_layout(
            title=dict(
                text=f"Interactive Waveform Viewer {filter_info}",
                x=0.5,
                y=0.95,
                font=dict(size=14)
            ),
            xaxis_title="Time (s)",
            yaxis_title="Voltage (V)", 
            height=700,
            margin=dict(t=80, r=300, b=80, l=80),  # Slightly larger right margin for enhanced legend
            hovermode='x unified'
        )

        # Configure enhanced individual trace legend
        fig = configure_individual_trace_legend_v10(fig)

        # Apply smart defaults for large datasets (respects enhanced button filtering)
        fig = apply_smart_default_visibility_v10(fig, collection, selection_filters)

        print(f"📊 Created interactive plot with {filtered_traces} traces (from {total_traces} total)")
        print(f"🔄 Individual trace control: Click legend entries")
        print(f"🎛️ Enhanced checkbox filtering: Use checkboxes above plot for precise wafer + parameter selection")
        print(f"🔍 Zoom in for full resolution, [R] indicators show resampled traces")

        return fig

    return (create_enhanced_resampled_plot_v10,)


@app.cell(hide_code=True)
def _(
    create_enhanced_resampled_plot_v10,
    g_selection_state_v10,
    tm_scope_collection_v10,
):
    # Create the enhanced resampled waveform plot with wafer support + push-button filtering (v10)
    print("🔄 Creating enhanced waveform visualization with wafer support + push-button filtering (v10)...")

    g_enhanced_plot_v10 = create_enhanced_resampled_plot_v10(
        tm_scope_collection_v10, 
        selection_filters=g_selection_state_v10
    )

    if tm_scope_collection_v10.data:
        print("✅ Enhanced plot v10 created successfully!")
        print("   🧬 Wafer selection: Filter by wafer type (Wave1Wafer3_A0-V1, Wave2WaferX_A1-V1, etc.)")
        print("   🎛️ Enhanced push-button filtering: Select WAFER + parameters to filter traces")
        print("   📊 Individual trace control: Each load condition selectable")
        print("   📋 Enhanced trace naming: Includes wafer identifier")
        print("   🔄 Automatic aggregation enabled (plotly-resampler)")
        print("   🔍 Zoom in for full resolution")
        print("   [R] indicators show resampled traces")
        print("   🎯 Smart defaults for large datasets")
        print("   💾 Memory optimized: <10MB output")

        if g_selection_state_v10.get("any_selected", False):
            print("   🎯 Enhanced button filters currently active (v10)")
    else:
        print("⚠️ No data available for visualization")

    return (g_enhanced_plot_v10,)


@app.cell(hide_code=True)
def _(g_checkbox_panel_v10, g_enhanced_plot_v10, mo):
    # Display the enhanced visualization UI with wafer checkbox controls positioned above plot (v10)
    g_visualization_ui_v10 = mo.vstack([
        mo.md("### 🎨 plotly_resampler.FigureResampler Waveform Visualization"),
        mo.md("""
        - **Wafer Checkbox Control:** Select specific wafer types for analysis
        - **Enhanced Dynamic Checkboxes:** WAFER, CH, VOUT, CBOOT, LOAD parameters populated from scope data extraction
        - **Independent Selection:** Each checkbox operates independently  
        - **Additive Filtering:** Multiple selections show multiple traces (e.g., Wave2WaferX_A1-V1 + SW + 0.8V)
        - **Unlimited Data:** Handles 7.5M+ data points without issue

        - **Checkboxes (below):** Check WAFER + parameters to filter displayed traces
        - **Legend (right):** Click traces to show/hide individual waveforms with wafer info
        - **Hover:** View detailed information including wafer identification
        - **Zoom/Pan:** Use mouse to explore waveform details (auto-loads full resolution)
        """),
        mo.md("---"),
        g_checkbox_panel_v10,
        mo.md("---"),
        g_enhanced_plot_v10
    ])

    return (g_visualization_ui_v10,)


@app.cell(hide_code=True)
def _(g_visualization_ui_v10):
    # Display the final enhanced visualization with wafer support (v10)
    g_visualization_ui_v10
    return


@app.cell
def _():
    # Suppress large outputs - all enhanced data processing complete (v10)
    pass
    return


if __name__ == "__main__":
    app.run()
