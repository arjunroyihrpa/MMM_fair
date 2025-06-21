# Author: Arjun Roy (arjun.roy@unibw.de, arjunroyihrpa@gmail.com) https://orcid.org/0000-0002-4279-9442
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import numpy as np
import pandas as pd
import fairbench as fb

from sklearn.tree import DecisionTreeClassifier

# local imports
from .data_process import data_uci
from .mammoth_csv import CSV
from .mmm_fair import MMM_Fair
from .mmm_fair_gb import MMM_Fair_GradientBoostedClassifier
from .deploy_utils import convert_to_onnx, convert_to_pickle
from .hyperparams import get_hparams  # The function that sets hyperparams or fallback
from sklearn.linear_model import LogisticRegression
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import train_test_split

from .fairlearn_report import generate_reports_from_fairlearn


# from mmm_fair.viz_trade_offs import plot3d

default_data_setting = {
    "bank": {"prots": ["marital", "age"], "nprotgs": ["married", "30_60"]},
    "adult": {"prots": ["race", "sex"], "nprotgs": ["White", "Male"]},
    # Add other dataset mappings here...
}


def html_report(views):
    tab_headers = "".join(
        f'<button class="tablinks" data-tab="{key}">{key}</button>' for key in views
    )
    tab_contents = "".join(
        f'<div id="{key}" class="tabcontent">{value}</div>'
        for key, value in views.items()
    )

    # dataset_desc = dataset.format_description()

    html_content = f"""
       <style>
           .tablinks {{
               background-color: #ddd;
               padding: 10px;
               cursor: pointer;
               border: none;
               border-radius: 5px;
               margin: 5px;
           }}
           .tablinks:hover {{ background-color: #bbb; }}
           .tablinks.active {{ background-color: #aaa; }}

           .tabcontent {{
               display: none;
               padding: 10px;
               border: 1px solid #ccc;
           }}
           .tabcontent.active {{ display: block; }}
       </style>
       <script>
           document.addEventListener("DOMContentLoaded", function() {{
               const tabContainer = document.querySelector("div");
               tabContainer.addEventListener("click", function(event) {{
                   if (event.target.classList.contains("tablinks")) {{
                       let tabName = event.target.getAttribute("data-tab");
                       document.querySelectorAll(".tablinks").forEach(tab => tab.classList.remove("active"));
                       document.querySelectorAll(".tabcontent").forEach(content => content.classList.remove("active"));
                       event.target.classList.add("active");
                       document.getElementById(tabName).classList.add("active");
                   }}
               }});

               // Show the first tab by default
               let firstTab = document.querySelector(".tablinks");
               if (firstTab) {{
                   firstTab.classList.add("active");
                   document.getElementById(firstTab.getAttribute("data-tab")).classList.add("active");
               }}
           }});
       </script>
       
       <div>{tab_headers}</div>
       <div>{tab_contents}</div>
       """
    return html_content


def generate_reports(
    report_type, sensitives, mmm_classifier, saIndex_test, y_pred, y_test, html=False
):
    """
    Generate pairwise fairness reports for different protected attributes.

    Parameters
    ----------
    report_type : Namespace or object with 'report_type' attribute.
    sensitives : list
        A list of the protected attribute names or identifiers.
    mmm_classifier : object
        An object that has a 'sensitives' attribute (e.g., a trained model
        with stored sensitive attribute information).
    saIndex_test : ndarray
        A 2D NumPy array (or similar) of sensitive attribute index data
        for the test set. Shape: (n_samples, n_protected_attributes).
    y_pred : array-like
        Predictions for the test set.
    y_test : array-like
        True labels for the test set.

    Returns
    -------
    string
    """
    # Decide which reporting type to use

    if html:
        rt = fb.export.HtmlTable  # (horizontal=False, view=False)
        views = {}
    else:
        if report_type.lower() == "table":
            rt = fb.export.ConsoleTable
        elif report_type.lower() == "console":
            rt = fb.export.Console

        else:
            print(
                f"Report type '{report_type}' not supported in this version. "
                "Switching to table type reporting."
            )
            rt = fb.export.ConsoleTable

    # Generate and display reports for each protected attribute
    out = ""
    for i in range(len(sensitives)):
        print(
            "Reports generated for protected attribute:", mmm_classifier.sensitives[i]
        )
        # Convert sensitive attribute column into a Fairlearn-friendly format
        sens = fb.categories(saIndex_test[:, i])

        # Create the fairness report
        report = fb.reports.pairwise(predictions=y_pred, labels=y_test, sensitive=sens)

        # Show/print the report
        if html:
            views[mmm_classifier.sensitives[i]] = report.show(
                env=fb.export.HtmlTable(view=False, filename=None)
            )
            # out += report.show(env=fb.export.Html(view=False))
        else:
            report.show(env=rt)
    if html:
        out = html_report(views)
    return out


def get_mmm_model(classifier="MMM_Fair", params={}):
    if classifier.lower() in ["mmm_fair", "mmm-fair", "mmm"]:
        return MMM_Fair(**params)
    elif classifier.lower() in ["mmm_fair_gbt", "mmm-fair-gbt", "mmm_gbt", "mmm-gbt"]:
        return MMM_Fair_GradientBoostedClassifier(**params)
    else:
        raise ValueError("Unknown classifier")


def build_sensitives(df: pd.DataFrame, protected_cols: list, non_protected_vals: list):
    """
    Constructs saIndex and saValue for fairness analysis.

    :param df: A pandas DataFrame containing all relevant data.
    :param protected_cols: List of column names for protected attributes.
    :param non_protected_vals: A list (same length) of "non-protected" specs,
        which can be:
        - Single values (str, float, int) for categorical or exact numeric matching,
        - A (lower, upper) tuple for numeric range checks.
    :return: (saIndex, saValue)
        saIndex: a 2D NumPy array of shape (n_samples, len(protected_cols)).
                 Each column i is 1 if the row's value is "non-protected" for that attribute,
                 else 0 if "protected".
        saValue: a dictionary {protected_col: 0, ...} indicating the convention
                 that 0 is considered protected.
    """

    if len(protected_cols) != len(non_protected_vals):
        raise ValueError(
            f"Number of protected columns ({len(protected_cols)}) does not match "
            f"the number of non-protected values ({len(non_protected_vals)})."
        )

    # We'll store 1 if row is "non-protected" according to the definition, else 0
    saIndex = df[protected_cols].to_numpy(copy=True)  # shape (n_samples, #prot_cols)
    saValue = {
        col: 0 for col in protected_cols
    }  # By default, interpret 0 as "protected"

    # Iterate over each protected column with the corresponding "non-protected" definition
    for i, (col, nprot_val) in enumerate(zip(protected_cols, non_protected_vals)):
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            parsed_value = parse_numeric_input(nprot_val)
            # If numeric, see if nprot_val is a tuple => range, or single numeric => exact match
            if isinstance(parsed_value, tuple) and len(parsed_value) == 2:
                lower, upper = parsed_value
                # "Non-protected" if value is between lower and upper
                saIndex[:, i] = (
                    (saIndex[:, i].astype(float) > float(lower))
                    & (saIndex[:, i].astype(float) < float(upper))
                ).astype(int)
            else:
                # Single numeric => direct match
                val_float = float(parsed_value)
                saIndex[:, i] = (saIndex[:, i].astype(float) == val_float).astype(int)

        else:
            # Categorical => simple string (or object) match
            saIndex[:, i] = (saIndex[:, i] == nprot_val).astype(int)

    return saIndex, saValue


def parse_numeric_input(value_str):
    """
    Parses a numeric input which can be:
    - A single numeric value: "30" -> 30.0
    - A range of two values: "30_60" -> (30.0, 60.0)

    Returns either:
    - A tuple (lower, upper) if it's a range.
    - A single float/int if it's just one number.
    """
    try:
        if "_" in value_str:
            lower, upper = map(float, value_str.split("_"))
            return lower, upper  # Return as a tuple for range
        else:
            return float(value_str)  # Return single numeric value
    except ValueError:
        raise ValueError(
            f"Invalid numeric format '{value_str}'. Expected a single number '30' or a range '30_60'."
        )


# --- Processing logic ---


def parse_base_learner(learner_str):
    if learner_str.lower() in ("tree", "dt", "decisiontree", "decision_tree"):
        return DecisionTreeClassifier(max_depth=5, class_weight=None)
    elif learner_str.lower() in ("xtree", "extra", "extratree", "extra_tree"):
        return ExtraTreeClassifier(max_features="sqrt")
    elif learner_str.lower() in ("logistic", "logreg", "lr"):
        return LogisticRegression(max_iter=1000)
    # elif learner_str.lower() in ("mlp","nn"):
    #     return  MLPClassifier()
    else:
        raise ValueError(f"Unrecognized base_learner: {learner_str}")
#def get

def train(args):
    dataset_name = None

    if args.dataset is not None:
        dataset_name = args.dataset.lower()

    # 1. Load data
    if args.df is not None:
        data= args.df
    else:
        if dataset_name and dataset_name.endswith(".csv"):
            # -------------------------
            # Local CSV file fallback
            # -------------------------
            raw_df = pd.read_csv(args.dataset)
    
            # Minimal auto-detection of numeric vs. categorical
            numeric = [
                col for col in raw_df.columns if pd.api.types.is_numeric_dtype(raw_df[col])
            ]
            categorical = [
                col for col in raw_df.columns if col not in numeric and col != args.target
            ]
    
            # Create the CSV object from mammoth_csv.py
            # We assume user-supplied --target is in raw_df
            label = raw_df[args.target]
    
            data = CSV(
                raw_df,
                numeric=numeric,
                categorical=categorical,
                labels=label,
            )
        else:
            # -------------------------
            # Known dataset (Adult, Bank, etc.)
            # via data_uci function
            # -------------------------
            data = data_uci(dataset_name=args.dataset, target=args.target)

    # 2. Retrieve hyperparameters & fallback for unknown data

    # -- Validate the prot_cols and nprotg_vals lengths match
    if len(args.prots) != len(args.nprotgs):
        raise ValueError(
            f"Number of protected attributes ({len(args.prots)}) "
            f"doesn't match number of non-protected values ({len(args.nprotgs)}). "
            f"Please provide them in pairs."
        )

    sensitives = []

    for col_index, (col, val) in enumerate(zip(args.prots, args.nprotgs)):
        # **Step 1: Check if col is in dataset, replace if needed**
        if col not in data.data.columns:
            dataset_name = args.dataset.lower()

            if dataset_name in default_data_setting:
                default_prots = default_data_setting[dataset_name]["prots"]
                default_nprotgs = default_data_setting[dataset_name]["nprotgs"]

                # Find a relevant protected attribute not already in args.prots
                for default_col, default_nprot in zip(default_prots, default_nprotgs):
                    if (
                        default_col not in args.prots
                    ):  # Only use if it's not already in the user's input
                        col = default_col  # Replace missing column with a relevant one
                        val = default_nprot  # Load corresponding nprotgs
                        print(
                            f"DEBUG: Replaced missing protected attribute '{args.prots[col_index]}' with default '{col}', using '{val}' as non-protected value."
                        )
                        args.prots[col_index] = col  # Update the list
                        args.nprotgs[col_index] = val  # Update nprotgs accordingly

                        break  # Stop searching once we find a replacement

            else:
                # If dataset is not known in default_data_setting, fall back to first categorical column
                categorical_cols = data.data.select_dtypes(
                    include=["object", "category"]
                ).columns
                if len(categorical_cols) == 0:
                    raise ValueError(
                        "No categorical columns available to replace the missing protected attribute."
                    )

                col = categorical_cols[0]  # Replace with first categorical column
                args.prots[col_index] = col  # Update the list with the new column name
                print(f"DEBUG: Replaced missing protected attribute with '{col}'.")

        sensitives.append(col)

        # **Step 2: Process numerical columns**
        if pd.api.types.is_numeric_dtype(data.data[col]):
            parsed_value = parse_numeric_input(val)
            if isinstance(parsed_value, tuple):
                if (
                    parsed_value[0] < data.data[col].min()
                    or parsed_value[1] > data.data[col].max()
                ):
                    raise ValueError(
                        f"{col} range '{val}' is outside dataset range [{data.data[col].min()}, {data.data[col].max()}]."
                    )
            else:  # If it's a single numeric value
                if (
                    parsed_value < data.data[col].min()
                    or parsed_value > data.data[col].max()
                ):
                    raise ValueError(
                        f"Numeric value '{val}' is outside dataset range [{data.data[col].min()}, {data.data[col].max()}]."
                    )

        # **Step 3: Process categorical columns**
        else:
            unique_vals = data.data[col].unique()
            if val not in unique_vals:
                if len(unique_vals) == 0:
                    raise ValueError(
                        f"No unique values found in column '{col}'. Cannot replace '{val}'."
                    )

                val = unique_vals[0]  # Replace with first unique value
                print(
                    f"DEBUG: Replaced '{args.nprotgs[col_index]}' with first unique value '{val}' in column '{col}'."
                )
                args.nprotgs[col_index] = val  # Update the list with the new value

    mmm_params, _ = get_hparams(
        classifier=args.classifier,
        dataset_name=args.dataset,
        constraint=args.constraint,
        data=data,
    )

    # mmm_params["saIndex"] = saIndex
    # mmm_params["saValue"] = saValue

    if args.base_learner is not None and args.classifier.lower() not in [
        "mmm_fair_gbt",
        "mmm-fair-gbt",
        "mmm_gbt",
        "mmm-gbt",
    ]:
        print(f"Loading MMM-Fair with base learner: {args.base_learner}")
        mmm_params["estimator"] = parse_base_learner(args.base_learner)
        if (
            isinstance(args.n_learners, (str, int))
            and str(args.n_learners).lstrip("-").isdigit()
        ):
            mmm_params["n_estimators"] = int(args.n_learners)
    else:
        if (
            isinstance(args.n_learners, (str, int))
            and str(args.n_learners).lstrip("-").isdigit()
        ):
            mmm_params["max_iter"] = int(args.n_learners)
        if args.early_stop == True:
            mmm_params["early_stopping"] = True
    # 3. Convert label array if needed
    y = data.labels["label"].to_numpy()
    if args.dataset.lower() == "adult":
        # Just an example if you want to transform e.g. "."
        y = np.array([s.replace(".", "") for s in y])
        # Possibly recast to 0/1 if you want:
    pos_class = args.pos_Class
    if pos_class not in list(set(y)):
        pos_class = list(set(y))[0]
    y = (y == pos_class).astype(int)

    # 4. Get feature matrix (some users do data.to_pred([...]) or data.to_features([...]))
    X = data.to_pred(sensitive=sensitives)  # or however you define

    # mmm_classifier = MMM_Fair(**mmm_params)

    # 5. Check if test data or split given
    if args.test is not None:
        # Could be a file or a float
        try:
            split_frac = float(args.test)
            if split_frac <= 0 or split_frac >= 1:
                raise ValueError("Train/Test split fraction must be between 0 and 1.")
            indices = np.arange(len(X))
            X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
                X, y, indices, test_size=split_frac, random_state=42, stratify=y
            )

            saIndex_train, saValue_train = build_sensitives(
                data.data.iloc[id_train], args.prots, args.nprotgs
            )
            saIndex_test, _ = build_sensitives(
                data.data.iloc[id_test], args.prots, args.nprotgs
            )

            mmm_params["saIndex"] = saIndex_train
            mmm_params["saValue"] = saValue_train

            # 6. Construct MMM_Fair
            mmm_classifier = get_mmm_model(
                classifier=args.classifier, params=mmm_params
            )  # MMM_Fair(**mmm_params)
            mmm_classifier.fit(X_train, y_train)

        except ValueError:
            # If it's not a float, treat it as a CSV path
            test_df = pd.read_csv(args.test)
            # Minimal numeric/categorical detection, build CSV instance
            numeric_test = [
                col
                for col in test_df.columns
                if pd.api.types.is_numeric_dtype(test_df[col])
            ]
            cat_test = [
                col
                for col in test_df.columns
                if col not in numeric_test and col != args.target
            ]
            label_test = test_df[args.target]

            data_test = CSV(
                test_df, numeric=numeric_test, categorical=cat_test, labels=label_test
            )

            if not list(data.data.columns) == list(data_test.data.columns):
                raise ValueError(
                    "Mismatch between train and test columns!\n"
                    f"Train columns: {list(data.data.columns)}\n"
                    f"Test columns: {list(data_test.data.columns)}"
                )
            X_test = data_test.to_pred(sensitive=sensitives)
            y_test = data_test.labels["label"].to_numpy()
            # y_pred = mmm_classifier.predict(X_test)
            # y_true=y_test
            # Train on entire main data
            saIndex, saValue = build_sensitives(data.data, args.prots, args.nprotgs)
            saIndex_test, _ = build_sensitives(data_test.data, args.prots, args.nprotgs)

            mmm_params["saIndex"] = saIndex
            mmm_params["saValue"] = saValue

            # 6. Construct MMM_Fair
            mmm_classifier = get_mmm_model(
                classifier=args.classifier, params=mmm_params
            )

            mmm_classifier.fit(X, y)
    else:
        saIndex, saValue = build_sensitives(data.data, args.prots, args.nprotgs)
        saIndex_test = saIndex
        mmm_params["saIndex"] = saIndex
        mmm_params["saValue"] = saValue
        # 6. Construct MMM_Fair
        mmm_classifier = get_mmm_model(classifier=args.classifier, params=mmm_params)
        mmm_classifier.fit(X, y)
        # test on the training data
        X_test, y_test = X, y

    # 6. Pareto setting
    mmm_classifier.pareto = args.pareto
    # if args.classifier not in ["mmm_fair_gbt","mmm-fair-gbt","mmm_gbt", "mmm-gbt"]:
    mmm_classifier.update_theta(criteria="all")


    baseline_results = None
    if hasattr(args, 'baseline_models') and args.baseline_models:
        print(f"\nTraining {len(args.baseline_models)} baseline models...")
        
        # Rest of baseline training logic using args.baseline_models
        if args.test is not None:
            try:
                split_frac = float(args.test)
                baseline_results = train_baseline_models(
                    args.baseline_models, X_train, y_train, X_test, y_test, saIndex_test, sensitives
                )
            except ValueError:
                baseline_results = train_baseline_models(
                    args.baseline_models, X, y, X_test, y_test, saIndex_test, sensitives
                )
        else:
            baseline_results = train_baseline_models(
                args.baseline_models, X, y, X, y, saIndex_test, sensitives
            )


    return mmm_classifier, X_test, y_test, saIndex_test, sensitives, baseline_results

    # 7. (Optional) FairBench reporting

    # If you only want the first protected col, do e.g. saIndex[:,0]
    # or otherwise combine them


def report_card(
    args, mmm_classifier, SI, sensitives, xtest, ytest, card=True, html=False
):
    ypred = mmm_classifier.predict(xtest)
    if card:
        if args.report_engine == 'fairbench':
            report = generate_reports(
                report_type=args.report_type,
                sensitives=sensitives,
                mmm_classifier=mmm_classifier,
                saIndex_test=SI,
                y_pred=ypred,
                y_test=ytest,
                html=html,
            )
        else:
            report = generate_reports_from_fairlearn(
                report_type=args.report_type,
                sensitives=sensitives,
                mmm_classifier=mmm_classifier,
                saIndex_test=SI,
                y_pred=ypred,
                y_test=ytest,
            )
    if args.moo_vis:
        mmm_classifier.see_pareto()
        while True:
            user_choice = input(
                "\nIf you wish to update the Model:\n"
                "Enter the Theta index (e.g., 0, 1, 2...) you chose from the Pareto plots,\n"
                "or enter 'exit' to keep the current model and exit: "
            ).strip()

            if user_choice.lower() == "exit":
                print("Exiting with the current model without updating further.")
                break  # Exit the loop

            try:
                theta = int(user_choice)
                if (
                    0 <= theta < len(mmm_classifier.ob)
                ):  # Ensure theta is within valid range
                    mmm_classifier.update_theta(theta=theta)
                    print(f"Model updated with Theta index {theta}.")
                    ypred = mmm_classifier.predict(xtest)
                    if args.report_engine == 'fairbench':
                        report = generate_reports(
                            report_type=args.report_type,
                            sensitives=sensitives,
                            mmm_classifier=mmm_classifier,
                            saIndex_test=SI,
                            y_pred=ypred,
                            y_test=ytest,
                            html=html,
                        )
                    else:
                        report = generate_reports_from_fairlearn(
                            report_type=args.report_type,
                            sensitives=sensitives,
                            mmm_classifier=mmm_classifier,
                            saIndex_test=SI,
                            y_pred=ypred,
                            y_test=ytest,
                        )
                    break  # Exit loop after successful update
                else:
                    print(
                        f"Invalid index! Please enter a valid Theta index (0 to {len(mmm_classifier.ob) - 1})."
                    )
            except ValueError:
                print(
                    "Invalid input! Please enter a valid integer Theta index or type 'exit' to quit."
                )

    return None


def deploy(stype, mmm_classifier, X, clf_name, path):
    # 8. Deployment
    if stype.lower() not in ("onnx", "pickle"):
        # If user didn't provide or gave something unrecognized => prompt
        while True:
            user_choice = input(
                "\nNo valid deployment option provided.\n"
                "Enter '1' for ONNX, '2' for pickle, '3' to exit: "
            ).strip()
            if user_choice == "1":
                stype = "onnx"
                break
            elif user_choice == "2":
                stype = "pickle"
                break
            elif user_choice == "3":
                print("Exiting without saving model.")
                return
            else:
                print("Invalid input. Please try again (1, 2, or 3).")

    # Now we have a recognized deploy format or user has chosen
    if stype.lower() == "onnx":
        convert_to_onnx(mmm_classifier, path, X, clf_name)
        print(f"Model saved in ONNX format with prefix '{path}'")
    elif stype.lower() == "pickle":
        convert_to_pickle(mmm_classifier, path)
        print(f"Model saved in pickle format as '{path}.pkl'")




def train_baseline_models(selected_models, X_train, y_train, X_test, y_test, saIndex_test, sensitives):
    """
    Train selected baseline models and return their results.
    
    Parameters:
    -----------
    selected_models : list
        List of baseline model names to train
    X_train, y_train : training data
    X_test, y_test : test data  
    saIndex_test : sensitive attribute indices for test data
    sensitives : list of sensitive attribute names
    
    Returns:
    --------
    dict : Results for each baseline model
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier  # Import MLP
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Define available baseline models
    baseline_model_map = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),  # Added MLP
    }
    
    results = {}
    
    for model_name in selected_models:
        if model_name in baseline_model_map:
            print(f"Training baseline model: {model_name}")
            
            # Get the model
            model = baseline_model_map[model_name]
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Generate fairness report
            # Create a mock classifier object with sensitives attribute for report generation
            class MockClassifier:
                def __init__(self, sensitives):
                    self.sensitives = sensitives
            
            mock_clf = MockClassifier(sensitives)
            
            # Generate HTML report
            report_html = generate_reports(
                report_type="html",
                sensitives=sensitives,
                mmm_classifier=mock_clf,
                saIndex_test=saIndex_test,
                y_pred=y_pred,
                y_test=y_test,
                html=True,
            )
            
            # Calculate basic metrics
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            fairobs = []

            for i in range(saIndex_test.shape[-1]):
                sens = fb.categories(saIndex_test[:, i])
                report = fb.reports.pairwise(
                    predictions=y_pred,
                    labels=y_test,
                    sensitive=sens
                )
                report_dict = report.show(env=fb.export.ToDict)
                
                dp = report_dict['depends'][6]['depends'][1]['value']['value']
                ep = report_dict['depends'][6]['depends'][2]['value']['value']
                tpr = report_dict['depends'][6]['depends'][2]['value']['value']
                fpr = report_dict['depends'][6]['depends'][3]['value']['value']
                eo = max(tpr, fpr)
                
                fairobs.append([dp, ep, eo, tpr, fpr])
                
            results[model_name] = {
                'model': model,
                'predictions': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'fairobs': fairobs,
                'fairness_report': report_html,
            }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train and Deploy MMM_Fair model")

    parser.add_argument(
        "--classifier",
        type=str,
        default="MMM_Fair",
        help="One of MMM_Fair (for original adaptive boosting version) or MMM_Fair_GBT (for gradient boosted trees)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Name of dataset or path to a local CSV file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Label column if using known dataset or CSV.",
    )
    parser.add_argument(
        "--pos_Class",
        type=str,
        default=None,
        help="Positive class Label if using known dataset or CSV.",
    )
    parser.add_argument(
        "--n_learners",
        type=str,
        default=None,
        help="Number of estimators or maxiters for the ensemble.",
    )
    parser.add_argument(
        "--prots",
        nargs="+",
        default=[],
        help="List of protected attribute names (e.g. --prots race sex age).",
    )
    # Similarly, a list of non-protected values for each attribute:
    parser.add_argument(
        "--nprotgs",
        nargs="+",
        default=[],
        help="List of non-protected attribute values, matching order of --prots.",
    )

    parser.add_argument(
        "--constraint",
        type=str,
        default="EO",
        help="Fairness constraint: DP, EO, EP, TPR, or FPR.",
    )

    parser.add_argument(
        "--deploy",
        type=str,
        default=None,
        help="Deployment format: 'onnx' or 'pickle'.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="my_mmm_fair_model",
        help="Path prefix for saved model(s).",
    )
    parser.add_argument(
        "--base_learner",
        type=str,
        default="lr",
        help="Override the default estimator, e.g. 'tree', 'logistic', etc.",
    )
    parser.add_argument(
        "--report_type",
        type=str,
        default="table",
        help="Override the default report output, e.g. 'table', 'console', 'html', etc.",
    )
    parser.add_argument(
        "--report_engine",
        type=str,
        default="fairbench",
        help="Override the default report engine, e.g. 'fairbench', 'fairlearn', etc.",
    )
    parser.add_argument(
        "--pareto",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Set to True to select theta from ensembles with Pareto optimal solutions (default: False)",
    )

    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Either path to a test CSV or a float fraction (e.g. 0.3) for train/test split. If not provided, no separate testing is done.",
    )
    parser.add_argument(
        "--early_stop",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Early stopping criteria for the GBT model",
    )

    parser.add_argument(
        "--moo_vis",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Set to True to visualize the Multi-objective plots solutions (default: False)",
    )
    parser.add_argument(
        "--df",
        default=None,)
    
    args = parser.parse_args()

    mmm_classifier, X_test, y_test, saIndex_test, sensitives, baseline_results = train(args)
    # y_pred = mmm_classifier.predict(X_test)
    plots = report_card(args, mmm_classifier, saIndex_test, sensitives, X_test, y_test)
    deploy(args.deploy, mmm_classifier, X_test, args.classifier, args.save_path)


if __name__ == "__main__":
    main()
