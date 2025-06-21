import os
import time
from flask import Flask, render_template, request, session, jsonify
from flask_session import Session
import argparse
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from flask import send_from_directory
# Import required modules
from mmm_fair.train_and_deploy import (
    generate_reports, 
    parse_base_learner,
    parse_numeric_input,
    build_sensitives,
    train
)
from mmm_fair.deploy_utils import convert_to_onnx
from mmm_fair.viz_trade_offs import plot2d, plot3d
from mmm_fair.mmm_fair import MMM_Fair
from mmm_fair.mmm_fair_gb import MMM_Fair_GradientBoostedClassifier
from mmm_fair.data_process import data_uci, data_local  # Importing data processing module
try:
    from mmm_fair.mchat.langchain_utils import (get_langchain_agent, 
                                        summarize_html_report, 
                                        get_available_llm_providers)
    LLM_AVAILABLE = True
except ImportError:
    print("🔕 LLM features not installed. To enable, use: pip install 'mmm-fair[llm-gpt]' or similar.")
    LLM_AVAILABLE = False
    
from mmm_fair.mchat.openai_utils import get_openAI_llm
from mmm_fair.mchat.groq_utils import get_groq_llm
from mmm_fair.mchat._utils import (filter_html, temp, remove_markdown, temp2, templates, 
                                    get_llm_context, summarize_data_distribution, 
                                    summarize_pareto)

from mmm_fair.viz_trade_offs import plot_spider
from mmm_fair.dataset_visualization import generate_nested_pie_chart

import uuid
import plotly.io as pio
llm=None

BASE_DIR = os.path.dirname(__file__)
PLOT_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(PLOT_DIR, exist_ok=True)

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=PLOT_DIR)
app.secret_key = "SOME_SECRET"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


THETA_DIV = """ 
<div id="theta-selection">
    <h3>Select Theta for Model Update</h3>
    <input type="number" id="theta-input" min="0" placeholder="Enter Theta index">
    <button id="theta-btn">Update Model</button>

    <h3>Save Model</h3>
    <input type="text" id="save-path" placeholder="Enter directory path">
    <button id="save-btn">Save Onnx</button>
</div>
"""

# Define chat arguments
CHAT_ARGS = [
    "dataset",
    "target",
    "prots",
    "nprotgs",
    "pos_Class",
    "data_visualization",
    "classifier",
    "constraint",
    "n_learners",
    "baseline_models",
    # "save_as",
    # "moo_vis"
]

default_args = {
    "classifier": "MMM_Fair_GBT",
    "dataset": "Adult",
    "target": "income",
    "pos_Class": ">50K",
    "n_learners": 100,
    "prots": ["race", "sex"],
    "nprotgs": ["White", "Male"],
    "constraint": "EO",
    "save_as": "Onnx",
    "save_path": "my_mmm_fair_model",
    "base_learner": None,
    "report_type": "table",
    "pareto": False,
    "test": 0.3,
    "early_stop": False,
    "moo_vis": True,
    "baseline_models": [],
}


# Dictionary of datasets and their recommended features
DATASET_RECOMMENDATIONS = {
    "adult": ["race", "sex", "age"],
    "bank": ["age", "marital", "education"],
    "credit": ["X2", "X4", "X5"],
    "kdd": ["AAGE", "ASEX", "ARACE"],
    "upload_data" : [],
}

TARGET_SUGGESTIONS = {
    "adult": [
        {"value": "income", "text": "income"},
    ],
    "bank": [
        {"value": "y", "text": "y (subscription)"},
    ],
    "credit": [
        {"value": "Y", "text": "default"},
    ],
    "kdd": [
        {"value": "income", "text": "income"},
    ],
    "upload_data" : [],
}

POS_CLASS_SUGGESTIONS = {
    "adult": {
        "income": [
            {"value": ">50K", "text": ">50K"},
            {"value": "<=50K", "text": "<=50K"},
        ],
    },
    "bank": {
        "y": [{"value": "yes", "text": "yes"}, {"value": "no", "text": "no"}],
    },
    "credit": {
        "default": [
            {"value": "1", "text": "1 (defaulted)"},
            {"value": "0", "text": "0 (not defaulted)"},
        ],
    },
    "kdd": {
        "income": [
            {"value": ' 50000+.', "text": ">50K$ (high income)"},
            {"value": '-50000', "text": "<50K$ (low income)"},
        ],
    },
    "upload_data" : {},
}

# Add this dictionary to your app.py file right after DATASET_RECOMMENDATIONS
NONPROTECTED_SUGGESTIONS = {
    "adult": {
        "race": [
            {"value": "White", "text": "White"},
            {"value": "Black", "text": "Black"},
            {"value": "Asian-Pac-Islander", "text": "Asian-Pac-Islander"},
        ],
        "sex": [
            {"value": "Male", "text": "Male"},
            {"value": "Female", "text": "Female"},
        ],
        "age": [
            {"value": "18_30", "text": "18-30"},
            {"value": "30_45", "text": "30-45"},
            {"value": "45_60", "text": "45-60"},
        ],
    },
    "bank": {
        "age": [
            {"value": "18_30", "text": "18-30"},
            {"value": "30_45", "text": "30-45"},
        ],
        "marital": [
            {"value": "married", "text": "Married"},
            {"value": "single", "text": "Single"},
        ],
        "education": [
            {"value": "primary", "text": "Primary"},
            {"value": "secondary", "text": "Secondary"},
            {"value": "tertiary", "text": "Tertiary"},
        ],
    },
    "credit": {
        "X2": [
            {"value": "1", "text": "1=Male"},
            {"value": "2", "text": "2=Female"},
        ],
        "X5": [
            {"value": "18_30", "text": "age: 18-30"},
            {"value": "30_45", "text": "age: 30-45"},
        ],
        "X4": [{"value": "1", "text": "1=married"}, {"value": "2", "text": "2=single"}, {"value": "3", "text": "3=others"}],
    },
    "kdd": {
        "ARACE": [
                    {"value": " White", "text": " White"},
                    {"value": " Asian or Pacific Islander", "text": " Asian or Pacific Islander"},
                    {"value": " Amer Indian Aleut or Eskimo", "text": " Amer Indian Aleut or Eskimo"},
                    {"value": " Black", "text": " Black"},
                        {"value": " Other", "text": " Other"}
                ],
        "ASEX": [{"value": ' Male', "text": "Male"}, {"value": ' Female', "text": "Female"}],
        "AAGE": [
            {"value": "18_30", "text": "18-30"},
            {"value": "30_45", "text": "30-45"},
            {"value": "45_60", "text": "45-60"},
            {"value": "60_80", "text": "Above 60"},
        ],
        
    },
    #"upload_data" : {},
}

BASELINE_MODEL_RECOMMENDATIONS = ["Decision Tree", "Random Forest", "MLP"]

# Dictionary to store available features for each dataset
DATASET_FEATURES = {}

# DEFAULT_BOT_MESSAGES = [
#     {
#         "sender": "bot",
#         "text": "Hello! Welcome to MMM-Fair Chat.\n\n"
#         "I can help you train either MMM_Fair (AdaBoost style) or MMM_Fair_GBT (Gradient Boosting style).\n"
#         "Please select an option below:",
#         "options": [
#             {"value": "MMM_Fair", "text": "MMM_Fair (AdaBoost)"},
#             {"value": "MMM_Fair_GBT", "text": "MMM_Fair_GBT (Gradient Boosting)"},
#             {"value": "default", "text": "Run with default parameters"},
#         ],
#     }
# ]
DEFAULT_BOT_MESSAGES = [
    {
        "sender": "bot",
        "text": "👋 Hello! Welcome to MMM-Fair Chat.\n\n"
                "Let's get started by selecting a dataset to work with.\n"
                "You can choose from well-known public datasets or upload your own CSV file.",
        "options": [
            {"value": "Adult", "text": "🧑 Adult (UCI)"},
            {"value": "Bank", "text": "🏦 Bank Marketing (UCI)"},
            {"value": "Credit", "text": "💳 Credit Default (UCI)"},
            {"value": "kdd", "text": "🧑 Census-Income KDD (UCI)"},
            {"value": "upload_data", "text": "📁 Upload your own Data (currently supported files: '.csv'"},
        ],
    }
]
@app.route("/")
def index():
    session["llm_enabled"] = False
    session["api_enabled"] = False
    return render_template("index.html")

@app.route("/upload_data", methods=["POST"])
def upload_csv():
    try:
        uploaded_file = request.files.get("file")
        if not uploaded_file or not uploaded_file.filename.endswith(".csv"):
            return jsonify({"success": False, "error": "Only CSV files are supported."})
        
        # Read and store dataframe in session
        df = pd.read_csv(uploaded_file)
        session["data"] = data_local(df) # Save to session
        #session["dataset_full"] = df
        return jsonify({"success": True, "columns": df.columns.tolist()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/provide_api_key", methods=["POST"])
def provide_api_key():
    try:
        env_key_map = {
                "openai": "OPENAI_API_KEY",
                "chatgpt": "OPENAI_API_KEY",
                "groq": "GROQ_API_KEY",
                "groqai": "GROQ_API_KEY",
                "together": "TOGETHER_API_KEY"
            }
        data = request.json
        api_key = data.get("api_key", "").strip()
        model = data.get("model", "").strip()
        if not api_key:
            return jsonify({"success": False, "error": "No API key provided."})

        if not model:
            return jsonify({"success": False, "error": "No LLM provider specified."})
        
        env_var = env_key_map.get(model.lower())
        if not env_var:
            return jsonify({"success": False, "error": f"Unknown provider '{model}'."})
            
        os.environ[env_var] = api_key
        session["api_enabled"] = True

        return jsonify({"success": True, "message": "✅ API key accepted. LLM features are now enabled."})

    except ImportError:
        return jsonify({
            "success": False,
            "error": "LangChain/OpenAI module not installed. Install with: pip install mmm-fair[llm-gpt]"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Unexpected error occurred: {str(e)}"
        })
    

@app.route("/start_chat", methods=["POST"])
def start_chat():
    session.clear()
    #session["chat_history"] = list(DEFAULT_BOT_MESSAGES)
    session["user_args"] = {}
    session["plot_all_url"]=''
    if session.get("llm_enabled"):      
        template=[("system","You are an assistant inside the MMM-Fair Chat."),
                  ("User", "Greet the user politely and explain that this tool can train fairness-aware models.Use the following context: {context}")]
        llm = get_openAI_llm()
        llm_agent=get_langchain_agent(llm,template)
        message = langchain_agent.invoke({"context": DEFAULT_BOT_MESSAGES[0]["text"]})
        session["chat_history"] = [{"sender": "bot", "text": message}]
    else:
        session["chat_history"] = list(DEFAULT_BOT_MESSAGES)
    return jsonify({"ok": True})


def prompt_for_llm_provider(chat_history):
    if LLM_AVAILABLE:
        available = get_available_llm_providers()
    else:
        available = LLM_AVAILABLE
    session["valid_providers"] = available
    if not available:
        chat_history.append({
            "sender": "bot",
            "text": "⚠️ No LLM providers are available. Please install with:\n"
                    "`pip install mmm-fair[llm-gpt]`, `[llm-groq]`, or `[llm-together]`"
        })
    else:
        session["llm_enabled"]=True

        available_options = [{"value": opt, "text": opt} for opt in list(available.keys())]
        
        chat_history.append({
            "sender": "bot",
            "text": f"🧠 Please select an available provider for explanation:",
            "options": available_options
        })
    return chat_history

def handle_llm_provider_selection(user_msg, chat_history):
    """Handle the LLM provider selection and set up for API key request."""
    user_selection = user_msg.lower()
    valid_providers = session.get("valid_providers")
    
    if user_selection in valid_providers:
        # Set the provider in the session
        session["llm_provider"] = user_selection
        
        # Set a flag to indicate we need an API key
        session["needs_api_key"] = True
        
        # Add a message to the chat history
        chat_history.append({
            "sender": "bot",
            "text": f"✅ Provider '{user_selection}' selected. Setting up for API key input..."
        })
        
        return chat_history, True  # Return True as second value to indicate API key needed
    else:
        chat_history.append({
            "sender": "bot",
            "text": f"⚠️ Unknown provider '{user_selection}'. Please choose one of: {', '.join(list(valid_providers.keys()))}"
        })
        
        return chat_history, False  # Return False to indicate no API key needed yet

@app.route("/ask_chat", methods=["POST"])
def ask_chat():
    chat_history = session.get("chat_history", [])
    user_args = session.get("user_args", {})
    user_args["df"] = None
    if "file" in request.files:
        user_msg = request.form.get("message", "").strip()
        selected_features = request.form.get(
            "selected_features", []
        )

        nprotgs_value = request.form.get("nprotgs_value", "")
        button_value = request.form.get("button_value", "")
        if button_value:
            user_msg = button_value
    else:
        
        user_msg = request.json.get("message", "").strip()
        button_value = request.json.get("button_value", "")
        selected_features = request.json.get(
            "selected_features", []
        )  # Get selected features if provided
    
        selected_models = request.json.get('selected_models', [])  # Add this line
        # Add this to handle non-protected group selection
        nprotgs_value = request.json.get("nprotgs_value", "")

    #print(f"DEBUG: Received button_value: {button_value}")
    #print("DEBUG: dataset =", user_args.get("dataset"))

    # If a button was clicked, use its value instead of the message
    if button_value:
        user_msg = button_value
        

    if (session.get("mmm_classifier") is not None and session.get("last_action") == "plotted" and user_msg.lower() in ["yes", "explain", "please explain", "what do these mean?", "openai", "chatgpt", "groqai", "groq", "togetherai", "together"]) or session.get("llm_context_active"):
        if not session.get("llm_enabled"):
            session["llm_provider"] = "in-progress"
            chat_history = prompt_for_llm_provider(chat_history)
            session["chat_history"] = chat_history
            
            return jsonify({"chat_history": chat_history[-1:]})
            
        elif session.get("llm_provider") and len(session.get("valid_providers", {})) > 0:
            if session.get("llm_provider") == "in-progress":
                # Process the LLM provider selection
                updated_history, needs_api_key = handle_llm_provider_selection(user_msg, chat_history)
                
                # Update the chat history
                chat_history = updated_history
                session["chat_history"] = chat_history
                
                # If we need an API key, indicate that to the frontend
                if needs_api_key:
                    return jsonify({
                        "chat_history": chat_history[-1:],
                        "require_api_key": True,
                        "provider": session.get("llm_provider")
                    })
                
                return jsonify({"chat_history": chat_history[-1:]})
                
            else:
                
                # Provider already selected, check if we need API key
                if not session.get("api_enabled"):
                    return jsonify({
                        "require_api_key": True, 
                        "provider": session.get("llm_provider"),
                        "chat_history": [{
                            "sender": "bot",
                            "text": f"Please provide your {session.get('llm_provider')} API key to continue."
                        }]
                    })
                
                # If we have API key, generate explanation
                try:
                    all_plots = session.get("plots")
                    mmm_classifier = session.get("mmm_classifier")
                    theta = mmm_classifier.theta
                    template = temp2
                    providers = session.get("valid_providers")
                    selected_model = session.get("llm_provider")
                    llm = providers[selected_model]()

                    
                    if session.get("llm_context_active"):
                        context= get_llm_context(user_msg)
                        print(f"Debug {context}")
                        summary = session.get("table_summary", {})
                        if context!="viz_pareto":
                            if context=="viz_fair":
                                template = [("system", templates["temp_exp"] + '\n' + templates["temp_fair"]), ("user",templates["temp_user"])]
                                llm_agent = get_langchain_agent(llm, template)
                                PF = np.array(
                [mmm_classifier.fairobs[i] for i in range(len(mmm_classifier.fairobs))])
                                obj_names =["DP", "EqOpp", "EqOdd", "TPR", "FPR"]
                                plot_html= summarize_pareto(PF, obj_names, 1)
                                
                            elif context=="viz_all":
                                template = [("system", templates["temp_exp"] + '\n' + templates["temp_all"]), ("user",templates["temp_user"])]
                                llm_agent = get_langchain_agent(llm, template)
                                PF = np.array([mmm_classifier.ob[i] for i in range(len(mmm_classifier.ob))])
                                obj_names = ["Acc. loss", "Balanc. Acc loss", "M3-fair loss"]
                                plot_html= summarize_pareto(PF, obj_names, 2)
                                
                            elif context=="viz_data":
                                template = [("system", templates["temp_exp"] + '\n' + templates["temp_data"]), ("user",templates["temp_user"])]
                                llm_agent = get_langchain_agent(llm, template)
                                data_obj = session.get("data")
                                user_args = session.get("user_args", {})
                                target = user_args.get("target")
                                protected_attrs = user_args.get("prots", [])
                                plot_html = summarize_data_distribution(data_obj, target, protected_attrs)
                                
                            else:  
                                template = [("system", templates["temp_exp"] + '\n' + templates["temp_tab"]), ("user",templates["temp_user"])]
                                llm_agent = get_langchain_agent(llm, template)
                                plot_html=all_plots[-1]['srcdoc']
                            #filtered_plot = filter_html(plot_html)
                            #print(f"Debug {plot_html[:1000]}")
                            summary_response = summarize_html_report(plot_html, llm_agent, summary="", question=user_msg) #summary.get("response", "")
                            summary_response = remove_markdown(summary_response)
                            if context=="summary":
                                summary["response"]= summary_response + f"\n this was the generated summary for the model with theta value {theta}"
                                summary["theta"]=theta
                                session["table_summary"]=summary
                                
                        elif context=="viz_pareto": 
                            template = [("system", templates["temp_exp"] + '\n' + templates["temp_all"] + '\n' + templates["temp_fair"]), ("user",templates["temp_user"])]
                            llm_agent = get_langchain_agent(llm, template)
                            PF = np.array(
                [mmm_classifier.fairobs[i] for i in range(len(mmm_classifier.fairobs))])
                            obj_names =["DP", "EqOpp", "EqOdd", "TPR", "FPR"]
                            summary_f= summarize_pareto(PF, obj_names, 1)
                            PF = np.array([mmm_classifier.ob[i] for i in range(len(mmm_classifier.ob))])
                            obj_names = ["Acc. loss", "Balanc. Acc loss", "M3-fair loss"]
                            summary_o= summarize_pareto(PF, obj_names, 1)
                            summary_response = summarize_html_report(summary_f + summary_o, llm_agent, summary="", question=user_msg)
                        
                    else:    
                        table_id = session.get("table_id", 0)
                        template = [("system", templates["temp_exp"] + '\n' + templates["temp_tab"]), ("user",templates["temp_user"])]
                        llm_agent = get_langchain_agent(llm, template)
    
                        table_plot = [item for item in all_plots if item.get("id") == table_id][0]
                        table_plot_html = table_plot["srcdoc"]
                        filtered_table = filter_html(table_plot_html)
    
                        summary_response = summarize_html_report(filtered_table, llm_agent)
                        summary_response = remove_markdown(summary_response)
                        
                        session["table_summary"]= {"response": summary_response + f"\n this was the generated summary for the model with theta value {theta}", "theta": theta}
                        session["llm_context_active"]=True
                        #session["agent"] = llm_agent#This is fatal as session cannot store such file
                    
                except Exception as e:
                    if "openai" in str(e).lower() or "key" in str(e).lower() or "quota" in str(e).lower():
                        return jsonify({
                            "require_api_key": True, 
                            "provider": session.get("llm_provider"),
                            "chat_history": [{
                                "sender": "bot",
                                "text": f"There was an issue with your API key. Please provide a valid {session.get('llm_provider')} API key."
                            }]
                        })
                
                    summary_response = f"(⚠️ Could not summarize the plot: {e})"
            
                summary_response += (
                            " \n\n If you have any further questions about the Pareto plots for fairness definitions or multi-objective trade-offs, data distribution, you can select from the options below or feel free to ask me through the chat.")  
                chat_history.append({"sender": "bot", "text": summary_response,
                        "options": [
                            {"value": "multi-objective", "text": "Multi-obs Pareto"},
                            {"value": "multi-definition", "text": "Multi-defs Pareto"},
                            {"value": "pareto", "text": "Best Theta"},
                            {"value": "data", "text": "Data Insights"},
                        ],})
                # Clean up session
                # if "valid_providers" in session:
                #     del session["valid_providers"]
                session["last_action"] = "summarized"
                session["chat_history"] = chat_history
                return jsonify({"chat_history": chat_history[-1:], "isMarkdown": True})
        
        elif session.get("llm_provider") == "in-progress" and len(session.get("valid_providers", {})) == 0:
            chat_history.append({
                "sender": "bot",
                "text": "You can also choose a Theta index below to update your model, and corresponding model's performance will also be updated. To start again with another Data/ Model, click on 'reset chat' 🙂"
            })
            return jsonify({"chat_history": chat_history[-1:]})
        
    # Handle non-protected group selection
    if user_msg.startswith("nprotg_") and nprotgs_value:
        # Initialize nprotgs if not exists
        if "nprotgs_temp" not in session:
            session["nprotgs_temp"] = []

        # Add the new value
        session["nprotgs_temp"].append(nprotgs_value)

        # Check if we have enough values
        if len(session["nprotgs_temp"]) >= len(user_args.get("prots", [])):
            # We have all values, combine them
            user_args["nprotgs"] = session["nprotgs_temp"]
            chat_history.append(
                {
                    "sender": "bot",
                    "text": f"Set nprotgs = {' '.join(session['nprotgs_temp'])}",
                }
            )

            # Clear temporary storage
            session.pop("nprotgs_temp", None)

            # Continue to next parameter
            new_missing = get_missing_args(user_args)
            if new_missing:
                next_arg = new_missing[0]
                prompt, options = get_prompt_for_arg(next_arg, user_args)
                chat_history.append(
                    {"sender": "bot", "text": prompt, "options": options}
                )
            else:
                chat_history.append(
                    {
                        "sender": "bot",
                        "text": "All arguments captured. What would you like to do?",
                        "options": [
                            {"value": "run", "text": "Run Training"},
                            {"value": "reset", "text": "Start Over"},
                        ],
                    }
                )

            session["chat_history"] = chat_history
            session["user_args"] = user_args
            return jsonify({"chat_history": chat_history[-2:]})
        else:
            # Still need more values, show progress
            remaining = len(user_args.get("prots", [])) - len(session["nprotgs_temp"])
            protected_attrs = user_args.get("prots", [])
            chat_history.append(
                {
                    "sender": "bot",
                    "text": f"Selected {nprotgs_value} for {protected_attrs[len(session['nprotgs_temp'])-1]}. "
                    + f"Please select {remaining} more value(s) for: {', '.join(protected_attrs[len(session['nprotgs_temp']):])}",
                }
            )

            # Show options for the next protected attribute
            next_prot = protected_attrs[len(session["nprotgs_temp"])]
            dataset_name = user_args.get("dataset", "").lower()

            if (
                dataset_name in NONPROTECTED_SUGGESTIONS
                and next_prot in NONPROTECTED_SUGGESTIONS[dataset_name]
            ):
                options = NONPROTECTED_SUGGESTIONS[dataset_name][next_prot]
                chat_history.append(
                    {
                        "sender": "bot",
                        "text": f"Select non-protected value for {next_prot}:",
                        "options": [
                            {
                                "value": f"nprotg_{next_prot}_{option['value']}",
                                "text": option["text"],
                            }
                            for option in options
                        ],
                    }
                )
            elif dataset_name == "upload_data":
                    options = [
                        {"value": str(val), "text": str(val)}
                        for val in sorted(set(session["data"].data[next_prot]))
                    ]
                    chat_history.append(
                        {
                            "sender": "bot",
                            "text": f"Select non-protected value for {next_prot}:",
                            "options": [
                                {
                                    "value": f"nprotg_{next_prot}_{val['value']}",
                                    "text": val["text"],
                                }
                                for val in options
                            ],
                        }
                    )

            session["chat_history"] = chat_history
            return jsonify({"chat_history": chat_history[-2:]})
            

    # Handle feature selection submission
    if user_msg == "submit_features" and selected_features:
        # Process submitted features
        if not selected_features:
            chat_history.append(
                {
                    "sender": "bot",
                    "text": "Error: Please select at least one protected attribute.",
                }
            )
        else:
            user_args["prots"] = selected_features.copy()
            chat_history.append(
                {
                    "sender": "bot",
                    "text": f"Set protected attributes = {', '.join(selected_features)}",
                }
            )

            # Continue to next parameter
            new_missing = get_missing_args(user_args)
            if new_missing:
                next_arg = new_missing[0]
                prompt, options = get_prompt_for_arg(next_arg, user_args)

                message = {"sender": "bot", "text": prompt}
                if options:
                    message["options"] = options

                chat_history.append(message)
            else:
                chat_history.append(
                    {
                        "sender": "bot",
                        "text": "All arguments captured. What would you like to do?",
                        "options": [
                            {"value": "run", "text": "Run Training"},
                            {"value": "reset", "text": "Start Over"},
                        ],
                    }
                )

            session["chat_history"] = chat_history
            session["user_args"] = user_args
            return jsonify({"chat_history": chat_history[-2:]})

    # Handle baseline models selection submission
    if user_msg == "submit_baseline_models" and selected_features:
        # Process submitted baseline models
        if not selected_features:
            chat_history.append(
                {
                    "sender": "bot",
                    "text": "No baseline models selected. Continuing without baseline comparison.",
                }
            )
            user_args["baseline_models"] = []
        else:
            user_args["baseline_models"] = selected_features.copy()
            chat_history.append(
                {
                    "sender": "bot",
                    "text": f"Selected baseline models: {', '.join(selected_features)}",
                }
            )

        # Continue to next parameter
        new_missing = get_missing_args(user_args)
        if new_missing:
            next_arg = new_missing[0]
            prompt, options = get_prompt_for_arg(next_arg, user_args)

            message = {"sender": "bot", "text": prompt}
            if options:
                message["options"] = options

            chat_history.append(message)
        else:
            chat_history.append(
                {
                    "sender": "bot",
                    "text": "All arguments captured. What would you like to do?",
                    "options": [
                        {"value": "run", "text": "Run Training"},
                        {"value": "reset", "text": "Start Over"},
                    ],
                }
            )

        session["chat_history"] = chat_history
        session["user_args"] = user_args
        return jsonify({"chat_history": chat_history[-2:]})

    if user_msg == "visualize_yes":
        # Store in user_args so we don't ask again
        user_args["data_visualization"] = "show_visualization"
        session["user_args"] = user_args
        # User wants to visualize data
        return jsonify({"redirect": "/visualize_data"})
    elif user_msg == "visualize_no":
        # Store in user_args so we don't ask again
        user_args["data_visualization"] = "skip_visualization"
        session["user_args"] = user_args
        # User doesn't want visualization, proceed to next step
        new_missing = get_missing_args(user_args)
        if new_missing:
            next_arg = new_missing[0]
            prompt, options = get_prompt_for_arg(next_arg, user_args)

            message = {"sender": "bot", "text": prompt}
            if options:
                message["options"] = options

            chat_history.append(message)
        else:
            chat_history.append(
                {
                    "sender": "bot",
                    "text": "All arguments captured. What would you like to do?",
                    "options": [
                        {"value": "run", "text": "Run Training"},
                        {"value": "reset", "text": "Start Over"},
                    ],
                }
            )

        session["chat_history"] = chat_history
        return jsonify({"chat_history": chat_history[-1:]})

    # **NEW FEATURE**: If user chooses "default", skip everything and run with default_args
    if not user_args and user_msg.lower() == "default":
        for arg in default_args:
            if arg not in user_args:
                user_args[arg] = default_args[arg]
        session["classifier"] = user_args["classifier"]
        chat_history.append(
            {"sender": "bot", "text": "Running MMM_Fair with default parameters..."}
        )
        visualizations, html_divs = run_mmm_fair_app(default_args)

        session["last_action"] = "plotted"

        default_resp="✅ Training complete! Here in the generated pareto plots (upper box) you see various solution points (models) available, where each solution point (denoted by theta: number) shows a different trade-off point between the different training objectives. \n\n The performance of the suggested best model is also plotted (lower box). You can also choose a Theta index below to update your model, and corresponding model's performance will also be updated."

        return jsonify(
            {
                "chat_history": [
                    {
                        "sender": "bot",
                        "text": f"{default_resp}\n\n If you'd like me to explain what you're seeing?.\n",
                        "options": [
                            {"value": "explain", "text": "Yes, please explain"},
                            {"value": "reset", "text": "Start Over"},
                        ],
                    }
                ],
                "plots": visualizations,
                "html_divs": html_divs,
            }
        )

    # Add user message
    if user_msg:
        # For button clicks, show the button text instead of value
        display_text = user_msg
        if chat_history:  # Check if chat_history is not empty
            options = chat_history[-1].get("options")
            if isinstance(options, list):
                for option in options: #chat_history[-1].get("options", []):
                    if option["value"] == user_msg:
                        display_text = option["text"]
                        break

        chat_history.append({"sender": "user", "text": display_text})

    missing_args = get_missing_args(user_args)

    if not missing_args:
        if user_msg.lower() == "run":
            for arg in default_args:
                if arg not in user_args:
                    user_args[arg] = default_args[arg]
            session["classifier"] = user_args["classifier"]
            visualizations, html_divs = run_mmm_fair_app(user_args)

            session["last_action"] = "plotted"
            default_resp="✅ Training complete! Here in the generated pareto plots (upper box) you see various solution points (models) available, where each solution point (denoted by theta: number) shows a different trade-off point between the different training objectives. \n\n The performance of the suggested best model is also plotted (lower box). You can also choose a Theta index below to update your model, and corresponding model's performance will also be updated."

            return jsonify(
                {
                    "chat_history": [
                        {
                            "sender": "bot",
                            "text": f"{default_resp}\n\n If you'd like me to explain what you're seeing?.\n",
                            "options": [
                                {"value": "explain", "text": "Yes, please explain"},
                                {"value": "reset", "text": "Start Over"},
                            ],
                        }
                    ],
                    "plots": visualizations,
                    "html_divs": html_divs,

                }
            )
        else:
            chat_history.append(
                {
                    "sender": "bot",
                    "text": "We have all arguments. What would you like to do?",
                    "options": [
                        {"value": "run", "text": "Run Training"},
                        {"value": "reset", "text": "Start Over"},
                    ],
                }
            )
    else:
        current_arg = missing_args[0]
        print(missing_args)
        valid, clean_val, err_msg = validate_arg(current_arg, user_msg, user_args)

        # Process classifier selection (from buttons)
        if current_arg == "classifier":
            if user_msg in ["MMM_Fair", "MMM_Fair_GBT"]:
                user_args[current_arg] = user_msg
                valid = True
                clean_val = user_msg

        # **NEW FEATURE**: Load dataset after fetching "dataset" argument
        if current_arg == "dataset":
            #print("we entered here")
            if valid:
                # Load dataset and store for later use
                try:
                    if "file" in request.files:
                        uploaded_file = request.files.get("file")
                        #print(f" - Filename: {uploaded_file.filename}")
                        df = pd.read_csv(uploaded_file)
                        session["data"] = data_local(df)
                        user_args["dataset"] = "upload_data"
                    else:    
                        session["data"] = data_uci(clean_val)  # Load dataset
                        user_args[current_arg] = clean_val
                    session["user_args"] = user_args#clean_val
                    chat_history.append(
                        {"sender": "bot", "text": f"Set {current_arg} = {clean_val}."}
                    )
                    chat_history.append(
                        {"sender": "bot", "text": "Dataset loaded successfully."}
                    )

                except Exception as e:
                    chat_history.append(
                        {
                            "sender": "bot",
                            "text": f"Error loading dataset: {str(e)}\nPlease select a valid dataset.",
                            "options": [
                                {"value": "adult", "text": "Adult Dataset"},
                                {"value": "bank", "text": "Bank Dataset"},
                                {"value": "credit", "text": "Credit Dataset"},
                                {"value": "kdd", "text": "KDD Dataset"},
                                {"value": "upload_data", "text": "📁 Upload your own Data (currently supported files: '.csv'"},
                            ],
                        }
                    )
                    session["chat_history"] = chat_history
                    return jsonify({"chat_history": chat_history[-2:]})

            else:
                chat_history.append(
                    {
                        "sender": "bot",
                        "text": f"Error: {err_msg}\nPlease select a dataset:",
                        "options": [
                            {"value": "adult", "text": "Adult Dataset"},
                            {"value": "bank", "text": "Bank Dataset"},
                            {"value": "credit", "text": "Credit Dataset"},
                            {"value": "kdd", "text": "KDD Dataset"},
                            {"value": "upload_data", "text": "📁 Upload your own Data (currently supported files: '.csv'"},
                        ],
                    }
                )
                session["chat_history"] = chat_history
                return jsonify({"chat_history": chat_history[-2:]})

        # Special handling for protected attributes
        elif current_arg == "prots":
            # Instead of validating text input, we'll create the feature selector UI
            prompt, options = get_prompt_for_arg(current_arg, user_args)

            message = {"sender": "bot", "text": prompt}

            if options:
                message["options"] = options

            chat_history.append(message)
            session["chat_history"] = chat_history
            return jsonify({"chat_history": chat_history[-1:]})

        elif current_arg == "constraint":
            if not valid:
                chat_history.append(
                    {
                        "sender": "bot",
                        "text": f"Error: {err_msg}\nPlease select a fairness constraint:",
                        "options": [
                            {"value": "DP", "text": "Demographic Parity (DP)"},
                            {"value": "EP", "text": "Equal Precision (EP)"},
                            {"value": "EO", "text": "Equal Opportunity (EO)"},
                            {"value": "TPR", "text": "True Positive Rate (TPR)"},
                            {"value": "FPR", "text": "False Positive Rate (FPR)"},
                        ],
                    }
                )
                session["chat_history"] = chat_history
                return jsonify({"chat_history": chat_history[-2:]})
            else:
                user_args[current_arg] = clean_val
                chat_history.append(
                    {"sender": "bot", "text": f"Set {current_arg} = {clean_val}."}
                )

        elif current_arg == "data_visualization" and valid:
            user_args[current_arg] = clean_val

            if clean_val == "show_visualization":
                # Redirect to the visualization endpoint
                return jsonify({"redirect": "/visualize_data"})
            else:
                # Just set the value and continue to next argument
                chat_history.append(
                    {"sender": "bot", "text": "Continuing without data visualization."}
                )

                # Now process as usual for the next missing argument
                new_missing = get_missing_args(user_args)
                if new_missing:
                    next_arg = new_missing[0]
                    prompt, options = get_prompt_for_arg(next_arg, user_args)

                    message = {"sender": "bot", "text": prompt}
                    if options:
                        message["options"] = options

                    chat_history.append(message)
                else:
                    chat_history.append(
                        {
                            "sender": "bot",
                            "text": "All arguments captured. What would you like to do?",
                            "options": [
                                {"value": "run", "text": "Run Training"},
                                {"value": "reset", "text": "Start Over"},
                            ],
                        }
                    )

                session["chat_history"] = chat_history
                session["user_args"] = user_args
                return jsonify({"chat_history": chat_history[-2:]})

        else:
            # For other arguments, still process text input but provide buttons when possible
            if not valid:
                chat_history.append(
                    {
                        "sender": "bot",
                        "text": f"Error: {err_msg}\nPlease re-enter {current_arg}.",
                    }
                )
            else:
                user_args[current_arg] = clean_val
                chat_history.append(
                    {"sender": "bot", "text": f"Set {current_arg} = {clean_val}."}
                )

        new_missing = get_missing_args(user_args)
        if new_missing:
            next_arg = new_missing[0]
            prompt, options = get_prompt_for_arg(next_arg, user_args)

            message = {"sender": "bot", "text": prompt}
            if options:
                message["options"] = options

            chat_history.append(message)
        else:
            chat_history.append(
                {
                    "sender": "bot",
                    "text": "All arguments captured. What would you like to do?",
                    "options": [
                        {"value": "run", "text": "Run Training"},
                        {"value": "reset", "text": "Start Over"},
                    ],
                }
            )

    session["chat_history"] = chat_history
    session["user_args"] = user_args
    return jsonify({"chat_history": chat_history[-2:]})


@app.route("/update_model", methods=["POST"])
def update_model():
    # Get Theta value from the request
    data = request.json
    theta_value = int(data.get("theta", -1))

    # Retrieve trained classifier
    mmm_classifier = session.get("mmm_classifier")
    if not mmm_classifier:
        return jsonify(
            {"success": False, "error": "No trained model found! Run training first."}
        )

    # Validate Theta index
    if theta_value < 0 or theta_value >= len(mmm_classifier.ob):
        return jsonify(
            {
                "success": False,
                "error": f"Invalid Theta index. Please select between 0 and {len(mmm_classifier.ob) - 1}.",
            }
        )

    # Update the model with selected Theta
    mmm_classifier.update_theta(theta=theta_value)
    session["mmm_classifier"] = mmm_classifier  # Store updated model

    X_test = session.get("xtest")
    y_test = session.get("ytest")
    user_args = session.get("user_args", {})
    sensitives = session.get("sensitives")
    saIndex_test = session.get("saIndex_test")

    y_pred = mmm_classifier.predict(X_test)

    report_table = generate_reports(
        "html", sensitives, mmm_classifier, saIndex_test, y_pred, y_test, html=True
    )

    table_id = session.get("table_id", 0)

    print(f"DEBUG: Updating report table with ID: {table_id}")

    report_table_plot_dict = {
        "updated_data":
        {
            "srcdoc": report_table,
        },
        "existing_id": table_id
    }
    all_plots=session.get("plots")
    all_plots[-1]['srcdoc']=report_table
    session["plots"] = all_plots



    # unique_id = str(uuid.uuid4())[:4]
    # plot_table = "table_.html"
    # p2 = f"table_{unique_id}.html"
    # plot_table_path = os.path.join(PLOT_DIR, p2)  # plot_table)

    # with open(plot_table_path, "w") as f:
    #     f.write(report_table)
    #     f.flush()
    #     os.fsync(f.fileno())

    # session["plot_fair_url"] = f"/static/{p2}"
    return jsonify(
        {
            "success": True,
            "message": f"Model updated with Theta index {theta_value}.",
            # "plot_fair_url": f"/static/{plot_table}",
            "plots": [report_table_plot_dict],
            "chat_history": [
                {
                    "sender": "bot",
                    "text": f"Model updated with Theta index {theta_value}. Would you like to save this model?",
                    "options": [
                        {"value": "save_model", "text": "Save Model"},
                        {"value": "reset", "text": "Start Over"},
                    ],
                }
            ],
        }
    )


def get_missing_args(user_args):
    """
    Return the list of arguments we still need, considering
    if user chose MMM_Fair_GBT => skip base_learner
    """
    # chosen_data = user_args.get("data", "").lower()
    # if not chosen_data:
    #     return ["dataset"]

    needed_args = []
    for arg in CHAT_ARGS:
        if arg == "base_learner":
            chosen_classifier = user_args.get("classifier", "").lower()
            if not chosen_classifier:
                return ["classifier"]
            elif chosen_classifier in [
            "mmm_fair_gbt",
            "mmm-fair-gbt",
            ]:
                continue  # Skip base_learner for GBT models

        if arg == "data_visualization":
            has_target = "target" in user_args and user_args["target"]
            has_prots = "prots" in user_args and user_args["prots"]

            if has_target and has_prots and "data_visualization" not in user_args:
                needed_args.append(arg)
            continue  # Skip to next argument regardless

        if arg not in user_args:
            needed_args.append(arg)

    return needed_args


def validate_arg(arg_name, user_input, user_args):
    """
    Simple validator that can parse or store defaults.
    Returns (valid, clean_value, error_message).
    """
    print(arg_name, user_input, user_args)
    if arg_name == "dataset":
        if user_input.lower().endswith(".csv"):
            return True, user_input, ""
        elif user_input.lower() in ["adult", "bank", "kdd", "credit","upload_data"]:
            return True, user_input.lower(), "Loading Data...please wait!!!"
        
        else:
            return False, user_input, ""  # Fallback case
    

    elif arg_name == "n_learners":
        if not user_input.isdigit():
            return False, None, "n_learners must be an integer e.g. 100"
        return True, int(user_input), ""

    elif arg_name == "constraint":
        c = user_input.upper()
        if c not in ["DP", "EP", "EO", "TPR", "FPR"]:
            return False, None, "Constraint must be DP, EP, EO, TPR, or FPR."
        return True, c, ""

    elif arg_name == "classifier":
        val = user_input.lower()
        if val not in ["mmm_fair", "mmm_fair_gbt", "mmm-fair", "mmm-fair-gbt"]:
            return False, None, "Classifier must be 'MMM_Fair' or 'MMM_Fair_GBT'."
        return True, "MMM_Fair_GBT" if "gbt" in val else "MMM_Fair", ""

    elif arg_name == "target":
        # Extract the name of the target column
        available_target_name = (
            session["data"].labels["label"].name
        )  # Get the column name

        #print(
        #    f"DEBUG: Available target name: {available_target_name}"
        #)  # Debugging output

        return True, user_input, ""

    elif arg_name == "data_visualization":
        if user_input.lower() in ["visualize_yes", "yes", "y"]:
            # Instead of storing a simple value, store a flag to redirect
            # This will be handled specially in ask_chat
            return True, "show_visualization", ""
        else:
            return True, "skip_visualization", ""

    elif arg_name == "pos_Class":
        if not user_input:
            return True, None, ""  # Default None
        return True, user_input, ""

    elif arg_name == "prots":
        available_columns = session["data"].data.columns.tolist()
        available_columns = [v.lower() for v in available_columns]
        if not user_input:
            return False, [], "At least one protected attribute expected"
        else:
            k = ""
            for prot in user_input.split(" "):
                if prot.lower() not in available_columns:
                    chosen_dataset = user_args.get("dataset", "").lower()
                    if chosen_dataset not in ["adult", "bank"]:
                        return (
                            False,
                            None,
                            f"Invalid protected. Available list of attributes in the data: {available_columns}.",
                        )
                    else:
                        k += " " + prot + ","

            if k != "":
                return (
                    True,
                    user_input.split(),
                    f"Invalid protected attribute(s) {k} will be replaced with default known protected.",
                )

            return True, user_input.split(), ""

    elif arg_name == "nprotgs":
        # If we're validating from button selection, we already handled the count check
        if "nprotgs_temp" in session and len(session["nprotgs_temp"]) > 0:
            return True, user_input.split(), ""
    
        # Otherwise, default validation
        existing_prots = user_args.get("prots", [])
        splitted = user_input.split()
        if len(splitted) != len(existing_prots):
            return (
                False,
                None,
                f"Got {len(splitted)} non-protected vals for {len(existing_prots)} protected columns. Please match count.",
            )
    
        # ✅ Add this return for the normal case
        return True, splitted, ""
    #elif arg_name == "nprotgs":
        # # If we're validating from button selection, we already handled the count check
        # if "nprotgs_temp" in session and len(session["nprotgs_temp"]) > 0:
        #     return True, user_input.split(), ""

        # # Otherwise, default validation
        # existing_prots = user_args.get("prots", [])
        # splitted = user_input.split()
        # if len(splitted) != len(existing_prots):
        #     return (
        #         False,
        #         None,
        #         f"Got {len(splitted)} non-protected vals for {len(existing_prots)} protected columns. Please match count.",
        #     )

    elif arg_name == "deploy":
        val = user_input.lower()
        if val not in ["onnx", "pickle"]:
            return False, None, "Deployment must be 'onnx' or 'pickle'."
        return True, val, ""

    elif arg_name == "moo_vis":
        if user_input.lower() in ["true", "yes", "y"]:
            return True, True, ""
        else:
            return True, False, ""

    elif arg_name == "baseline_models":
        if not user_input:
            return True, [], ""  # Empty list is valid for baseline_models
        else:
            invalid_models = []
            for model in user_input.split():
                if model not in BASELINE_MODEL_RECOMMENDATIONS:
                    invalid_models.append(model)
            
            if invalid_models:
                return (
                    False,
                    None,
                    f"Invalid baseline model(s): {', '.join(invalid_models)}. Available models: {', '.join(BASELINE_MODEL_RECOMMENDATIONS)}",
                )
            
            return True, user_input.split(), ""

    else:
        return True, user_input, ""  # Default case


# Modified get_prompt_for_arg to provide dataset features
def get_prompt_for_arg(arg_name, user_args):
    """
    Return a question/prompt for the user and available button options if applicable
    based on arg_name
    """
    prompts = {
        "dataset": (
            "Please select a dataset:",
            [
                {"value": "adult", "text": "Adult Dataset"},
                {"value": "bank", "text": "Bank Dataset"},
                {"value": "credit", "text": "Credit Dataset"},
                {"value": "kdd", "text": "KDD Dataset"},
            ],
        ),
        "target": (
            "Enter the label (target) column in your dataset (e.g. 'income'):",
            None,
        ),
        "pos_Class": (
            "Enter the positive class label if known (else press Enter to skip):",
            None,
        ),
        "n_learners": (
            "How many learners / iterations?",
            [
                {"value": "50", "text": "50 Learners"},
                {"value": "100", "text": "100 Learners"},
                {"value": "200", "text": "200 Learners"},
            ],
        ),
        "constraint": (
            "Please select a fairness constraint:",
            [
                {"value": "DP", "text": "Demographic Parity (DP)"},
                {"value": "EP", "text": "Equal Opportunity (EqOpp)"},
                {"value": "EO", "text": "Equalized Odds (EqOdd)"},
                {"value": "TPR", "text": "True Positive Rate (TPR)"},
                {"value": "FPR", "text": "False Positive Rate (FPR)"},
            ],
        ),
        "nprotgs": (
            "Enter corresponding non-protected spec, e.g. 'White Male 30_60' matching the above columns:",
            None,
        ),
        "deploy": (
            "Would you like to deploy as 'onnx' or 'pickle'?",
            [
                {"value": "onnx", "text": "ONNX Format"},
                {"value": "pickle", "text": "Pickle Format"},
            ],
        ),
        "moo_vis": (
            "Do you want to enable multi-objective visualization?",
            [
                {"value": "True", "text": "Yes, enable visualization"},
                {"value": "False", "text": "No, disable visualization"},
            ],
        ),
        "data_visualization": (
            "Would you like to visualize the distribution of your data?",
            [
                {"value": "visualize_yes", "text": "Yes, show visualization"},
                {"value": "visualize_no", "text": "No, continue without visualization"},
            ],
        ),
    }

    if arg_name == "classifier":
        return (
            "Please select a model type:",
            [
                {"value": "MMM_Fair", "text": "MMM_Fair (AdaBoost)"},
                {"value": "MMM_Fair_GBT", "text": "MMM_Fair_GBT (Gradient Boosting)"},
            ],
        )

    if arg_name == "dataset":
        return  (
            "Please select a dataset:",
            [
                {"value": "adult", "text": "Adult Dataset"},
                {"value": "bank", "text": "Bank Dataset"},
                {"value": "credit", "text": "Credit Dataset"},
                {"value": "kdd", "text": "KDD Dataset"},
                {"value": "upload_data", "text": "📁 Upload your own Data (currently supported types: '.csv'"},
            {"value": "default", "text": "Run with default setup on Adult data"}
            ],
        )
        
    if arg_name == "base_learner":
        return (
            "Select a base learner:",
            [
                {"value": "tree", "text": "Decision Tree"},
                {"value": "lr", "text": "Linear Regression"},
                {"value": "logistic", "text": "Logistic Regression"},
                {"value": "extratree", "text": "Extra Tree"},
            ],
        )

    if arg_name == "prots":
        # Get dataset name
        dataset_name = user_args.get("dataset", "").lower()
        #print(f'Debugging inside get_prompt {dataset_name}')
        available_features = []
        recommended_features = []

        if "data" in session:
            #print("we came here for data")
            # Get actual columns from the dataset
            available_features = session["data"].data.columns.tolist()
            #print(f'Debugging inside get_prompt features {available_features}')
            DATASET_FEATURES[dataset_name] = available_features
        elif dataset_name in DATASET_FEATURES:
            # Use cached features if available
            available_features = DATASET_FEATURES[dataset_name]

        # Get recommended features for this dataset
        #recommended_features = []
        if dataset_name in DATASET_RECOMMENDATIONS:
            recommended_features = DATASET_RECOMMENDATIONS[dataset_name]

        #if dataset_name == "upload_data":
        #    recomended_features= available_features
        # Build a special message with recommendations
        message = "Please select protected attributes:"
        if recommended_features:
            message += f"\n\nRecommended for {dataset_name.capitalize()}: " + ", ".join(
                recommended_features
            )

        # Return a special format with available features
        
        print(f'Debugging for recomended values {recommended_features}')
        return (
            message,
            {
                "type": "features_selector",
                "available": available_features,
                "recommended": recommended_features,
                "selector_title": "Select Protected Attributes",
                "item_label": "feature"
            },
        )

    elif arg_name == "nprotgs":
        dataset_name = user_args.get("dataset", "").lower()
        protected_attrs = user_args.get("prots", [])
    
        # Case 1: Use default suggestions for known datasets
        if dataset_name in NONPROTECTED_SUGGESTIONS and protected_attrs:
            if "nprotgs_temp" in session and session["nprotgs_temp"]:
                next_idx = len(session["nprotgs_temp"])
                if next_idx < len(protected_attrs):
                    next_prot = protected_attrs[next_idx]
                    if next_prot in NONPROTECTED_SUGGESTIONS[dataset_name]:
                        options = NONPROTECTED_SUGGESTIONS[dataset_name][next_prot]
                        return (
                            f"Select non-protected value for {next_prot}:",
                            [
                                {
                                    "value": f"nprotg_{next_prot}_{option['value']}",
                                    "text": option["text"],
                                }
                                for option in options
                            ],
                        )
            elif protected_attrs:
                first_prot = protected_attrs[0]
                if first_prot in NONPROTECTED_SUGGESTIONS[dataset_name]:
                    options = NONPROTECTED_SUGGESTIONS[dataset_name][first_prot]
                    return (
                        f"Select non-protected value for {first_prot}:",
                        [
                            {
                                "value": f"nprotg_{first_prot}_{option['value']}",
                                "text": option["text"],
                            }
                            for option in options
                        ],
                    )
    
        # Case 2: Dynamically build dropdown from uploaded CSV
        if dataset_name == "upload_data" and "data" in session:
            df = session["data"].data
            if protected_attrs:
                temp_vals = session.get("nprotgs_temp", [])
                next_idx = len(temp_vals)
                if next_idx < len(protected_attrs):
                    next_prot = protected_attrs[next_idx]
                    try:
                        values = sorted(set(df[next_prot].dropna().unique()))
                        options = [
                            {
                                "value": f"nprotg_{next_prot}_{val}",
                                "text": str(val),
                            }
                            for val in values
                        ]
                        return (
                            f"Select non-protected value for '{next_prot}':",
                            options
                        )
                    except Exception as e:
                        return (
                            f"⚠️ Could not retrieve values for '{next_prot}': {e}",
                            None
                        )
            else:
                session["nprotgs_temp"] = []
    
        return (
            "Enter corresponding non-protected spec, e.g. 'White Male' matching the above columns:",
            None,
        )
    elif arg_name == "target":
        target=session['data'].labels['label'].name
        dataset_name = user_args.get("dataset", "").lower()
        if dataset_name in TARGET_SUGGESTIONS:
            if dataset_name == "upload_data":
                TARGET_SUGGESTIONS[dataset_name]= [{"value": target, "text": target},]
            return (
                "Select the label (target) column in your dataset:",
                TARGET_SUGGESTIONS[dataset_name],
            )
        else:
            target=session['data'].labels['label'].name
            print(f'Debugging target name{target}')
            return (
                "Enter the label (target) column in your dataset:",
                {"value": target, "text": target},
            )

    elif arg_name == "pos_Class":
        dataset_name = user_args.get("dataset", "").lower()
        target_column = user_args.get("target", "")
    
        try:
            if "data" in session and target_column:
                labels = session["data"].labels["label"]
                unique_classes = list(set(labels))
                options = [{"value": str(c), "text": f"Use '{c}' as positive"} for c in unique_classes]
    
                # Add a "Skip" option too
                options.insert(0, {"value": "", "text": "Skip (use default)"})
                return ("Please select the positive class label:", options)
    
        except Exception as e:
            print(f"DEBUG: Failed to fetch pos_Class options — {e}")
    
        return ("Enter the positive class label if known (else press Enter to skip):", None)

    elif arg_name == "baseline_models":
        return (
            "Please select baseline models for comparison (optional):",
            {
                "type": "features_selector",
                "available": BASELINE_MODEL_RECOMMENDATIONS,
                "recommended": BASELINE_MODEL_RECOMMENDATIONS,
                "selector_title": "Select Baseline Models",
                "item_label": "model",
                "arg_name": "baseline_models"
            },
        )

    return prompts.get(arg_name, (f"Enter {arg_name}:", None))


# Add a new route to handle feature selection
# Add a new route to handle feature selection
@app.route("/select_feature", methods=["POST"])
def select_feature():
    data = request.json
    feature = data.get("feature", "")
    selector_type = data.get("selector_type", "prots")  # Default to prots for backward compatibility

    # Get current selected features
    user_args = session.get("user_args", {})
    
    if selector_type == "baseline_models":
        current_items = user_args.get("baseline_models", [])
        key = "baseline_models"
    else:
        current_items = user_args.get("prots", [])
        key = "prots"

    # Add the feature if not already selected
    if feature and feature not in current_items:
        current_items.append(feature)
        user_args[key] = current_items
        session["user_args"] = user_args

    return jsonify({"success": True, "selected_features": current_items})

@app.route("/visualize_data", methods=["POST"])
def visualize_data():
    """Generate and return a visualization of the dataset distribution"""
    # Get data from session
    if "data" not in session:
        return jsonify(
            {
                "success": False,
                "error": "No dataset loaded. Please load a dataset first.",
            }
        )

    data_obj = session.get("data")
    user_args = session.get("user_args", {})

    # Get target and protected attributes
    target = user_args.get("target")
    protected_attrs = user_args.get("prots", [])

    if not target or not protected_attrs:
        return jsonify(
            {"success": False, "error": "Target or protected attributes not defined."}
        )

    try:
        # Generate the plot using your existing function
        df = data_obj.data.copy()  # Get the dataframe from your data object
        df[target] = data_obj.labels["label"].values.copy()
        # Generate a unique filename
        # unique_id = str(uuid.uuid4())[:8]
        # pie_plot_filename = f"data_dist_{unique_id}.html"
        # pie_plot_path = os.path.join(PLOT_DIR, pie_plot_filename)

        # Generate the nested pie chart
        data_plot_html = generate_nested_pie_chart(df, [target] + protected_attrs)

        
        style_dict = {  
            "width": "100%",
            "height": "500px",
            "overflow": "hidden",
            "border": "None",     
        }

        data_plot_dict = {
            "srcdoc": data_plot_html,
            "style": style_dict,
            "id": str(uuid.uuid4())[:4]
        }

        # print(plot_html)
        # # Save the plot
        # with open(pie_plot_path, "w") as f:
        #     f.write(plot_html)
        #     f.flush()
        #     os.fsync(f.fileno())
        #     # Get the next prompt to display after visualization

        # Mark visualization as completed in user_args
        session["data_viz"] = data_plot_dict
        user_args["data_visualization"] = "completed"
        session["user_args"] = user_args

        # Get the next prompt to display after visualization
        chat_history = session.get("chat_history", [])
        chat_history.append(
            {
                "sender": "bot",
                "text": "Here's the distribution of your data based on target and protected attributes.",
            }
        )

        # Determine the next argument to prompt for
        new_missing = get_missing_args(user_args)

        if new_missing:
            next_arg = new_missing[0]
            prompt, options = get_prompt_for_arg(next_arg, user_args)

            next_message = {"sender": "bot", "text": prompt}
            if options:
                next_message["options"] = options

            chat_history.append(next_message)
        else:
            # If no more arguments needed, show the run options
            chat_history.append(
                {
                    "sender": "bot",
                    "text": "All arguments captured. What would you like to do?",
                    "options": [
                        {"value": "run", "text": "Run Training"},
                        {"value": "reset", "text": "Start Over"},
                    ],
                }
            )

        session["chat_history"] = chat_history

        return jsonify(
            {
                "success": True,
                "plots": [data_plot_dict],  # Send HTML directly instead of file path
                "chat_history": chat_history[
                    -2:
                ],  # Return both visualization message and next prompt
            }
        )
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        return jsonify(
            {"success": False, "error": f"Error generating visualization: {str(e)}"}
        )


@app.route("/finish_baseline_models", methods=["POST"])
def finish_baseline_models():

    print(f"DEBUG: Selected features finish_baseline_models called")

    user_args = session.get("user_args", {})
    selected_models = user_args.get("baseline_models", [])

    chat_history = session.get("chat_history", [])

    # Add message showing selected models
    if selected_models:
        chat_history.append(
            {
                "sender": "bot",
                "text": f"Selected baseline models: {', '.join(selected_models)}",
            }
        )
    else:
        chat_history.append(
            {
                "sender": "bot",
                "text": "No baseline models selected. Continuing without baseline comparison.",
            }
        )

    # Continue to next parameter
    new_missing = get_missing_args(user_args)
    if new_missing:
        next_arg = new_missing[0]
        prompt, options = get_prompt_for_arg(next_arg, user_args)

        message = {"sender": "bot", "text": prompt}
        if options:
            message["options"] = options

        chat_history.append(message)
    else:
        chat_history.append(
            {
                "sender": "bot",
                "text": "All arguments captured. What would you like to do?",
                "options": [
                    {"value": "run", "text": "Run Training"},
                    {"value": "reset", "text": "Start Over"},
                ],
            }
        )

    session["chat_history"] = chat_history
    return jsonify({"success": True, "chat_history": chat_history[-2:]})

@app.route("/finish_features", methods=["POST"])
def finish_features():
    user_args = session.get("user_args", {})
    selected_features = user_args.get("prots", [])

    if not selected_features:
        return jsonify(
            {
                "success": False,
                "error": "Please select at least one protected attribute.",
            }
        )

    # Convert to the format expected by the existing code
    chat_history = session.get("chat_history", [])


    print(f"DEBUG: Selected features finished_features called")

    # Add message showing selected features
    chat_history.append(
        {
            "sender": "bot",
            "text": f"Selected protected attributes: {', '.join(selected_features)}",
        }
    )

    # Check if we have both target and protected attributes
    if "target" in user_args and user_args["target"]:
        # Add suggestion to visualize data
        chat_history.append(
            {
                "sender": "bot",
                "text": "Would you like to visualize the distribution of your data?",
                "options": [
                    {"value": "visualize_yes", "text": "Yes, show visualization"},
                    {
                        "value": "visualize_no",
                        "text": "No, continue without visualization",
                    },
                ],
            }
        )

        session["chat_history"] = chat_history
        return jsonify({"success": True, "chat_history": chat_history[-2:]})

    # If no target yet, continue with original flow
    # Continue to next parameter
    new_missing = get_missing_args(user_args)
    if new_missing:
        next_arg = new_missing[0]
        prompt, options = get_prompt_for_arg(next_arg, user_args)

        message = {"sender": "bot", "text": prompt}
        if options:
            message["options"] = options

        chat_history.append(message)
    else:
        chat_history.append(
            {
                "sender": "bot",
                "text": "All arguments captured. What would you like to do?",
                "options": [
                    {"value": "run", "text": "Run Training"},
                    {"value": "reset", "text": "Start Over"},
                ],
            }
        )

    session["chat_history"] = chat_history
    return jsonify({"success": True, "chat_history": chat_history[-2:]})


@app.route("/reset_chat")
def reset_chat():
    session.pop("chat_history", None)
    session.pop("user_args", None)
    session.pop("data", None)
    for fname in os.listdir(PLOT_DIR):
        if (
            fname.startswith("table_")
            or fname.startswith("fair_")
            or fname.startswith("all_")
        ):
            try:
                os.remove(os.path.join(PLOT_DIR, fname))
            except Exception:
                pass
    return "Chat reset done."

def run_baselines(user_args):
    """Train baseline models and return results"""
    try:
        selected_baselines = user_args.get("baseline_models", [])
        results = {}
        for model in selected_baselines:
            # Implement training for each baseline model
            # This is a placeholder - replace with actual implementation
            results[model] = f"Trained {model} successfully"
        
        return results
    except Exception as e:
        print(f"Error training baselines: {str(e)}")
        return {}


def run_mmm_fair_app(user_args):
    data = session.get('data')
    user_args["df"]=data
    args = argparse.Namespace(**user_args)
    mmm_classifier, X_test, y_test, saIndex_test, sensitives, baseline_results = train(args)


    session["mmm_classifier"] = mmm_classifier
    session["xtest"] = X_test
    session["ytest"] = y_test
    session["saIndex_test"] = saIndex_test
    session["sensitives"] = sensitives
    PF = np.array([mmm_classifier.ob[i] for i in range(len(mmm_classifier.ob))])
    thetas = np.arange(len(mmm_classifier.ob))
    title = f"2D Spider Plot. Showing various trade-off points between Accuracy, Balanced Accuracy, and Maximum violation of {mmm_classifier.constraint} fairness among protected attributes."

    vis_all = plot_spider(
        objectives=PF,
        theta=thetas,
        criteria="Multi",
        axis_names=["Acc. loss", "Balanc. Acc loss", "M3-fair loss"],
        title=title,
        html=True,
    )
    # vis_all = plot3d(
    #     x=PF[:, 0],
    #     y=PF[:, 1],
    #     z=PF[:, 2],
    #     theta=thetas,
    #     criteria="Multi",
    #     axis_names=["Acc.", "Balanc. Acc", "MMM-fair"],
    #     title=title,
    #     html=True,
    # )
    PF = np.array(
        [mmm_classifier.fairobs[i] for i in range(len(mmm_classifier.fairobs))]
    )
    title = "2D Spider Plot. Showing various trade-off points between maximum violation of Demopgraphic Parity, Equal Opportunity, and Equalized odds fairness for the given set of protected attributes."
    vis_fair = plot_spider(
        objectives=PF,
        theta=thetas,
        baseline_results=baseline_results,
        criteria="Multi-definitions",
        axis_names=["DP", "EqOpp", "EqOdd", "TPR", "FPR"],
        title=title,
        html=True,
    )

    # vis_fair = plot3d(
    #     x=PF[:, 0],
    #     y=PF[:, 1],
    #     z=PF[:, 2],
    #     theta=thetas,
    #     criteria="Multi-definitions",
    #     axis_names=["DP", "EqOpp", "EqOdd"],
    #     title=title,
    #     html=True,
    # )

    y_pred = mmm_classifier.predict(X_test)
    report_table = generate_reports(
        "html", sensitives, mmm_classifier, saIndex_test, y_pred, y_test, html=True
    )
    # plot_all = "all_.html"
    # plot_fair = "fair_.html"
    # plot_table = "table_.html"

    # plot_all_path = os.path.join(PLOT_DIR, plot_all)
    # plot_fair_path = os.path.join(PLOT_DIR, plot_fair)
    # plot_table_path = os.path.join(PLOT_DIR, plot_table)

    # # Save the Plotly-generated HTML directly to files
    # with open(plot_all_path, "w") as f:
    #     f.write(vis_all)
    # with open(plot_fair_path, "w") as f:
    #     f.write(vis_fair)
    # with open(plot_table_path, "w") as f:
    #     f.write(report_table)

    style_dict = {  
        "width": "100%",
        "height": "500px",
        "overflow": "hidden",
        "border": "None",     
    }

    vis_all_plot_dict = {
        "srcdoc": vis_all,
        "style": style_dict,
        "id": str(uuid.uuid4())[:4]
    }
    vis_fair_plot_dict = {
        "srcdoc": vis_fair,
        "style": style_dict,
        "id": str(uuid.uuid4())[:4]
    }    
    table_id = str(uuid.uuid4())[:4]
    report_table_plot_dict = {
        "srcdoc": report_table,
        "style": style_dict,
        "id": table_id
    }
    session["plots"] = [vis_all_plot_dict, vis_fair_plot_dict, report_table_plot_dict]
    session["table_id"] = table_id
    session["html_divs"] = [THETA_DIV]

    return [vis_all_plot_dict, vis_fair_plot_dict, report_table_plot_dict], [THETA_DIV]



@app.route("/static/<path:filename>")
def serve_static_files(filename):
    return send_from_directory("static", filename)


@app.route("/static/<path:filename>")
def serve_plot(filename):
    plot_dir = os.path.abspath("static/plots")  # Ensure absolute path
    return send_from_directory(plot_dir, filename)


@app.route("/save_model", methods=["POST"])
def save_model():
    data = request.json
    save_path = data.get("save_path", "").strip()

    # Retrieve trained classifier
    mmm_classifier = session.get("mmm_classifier")
    clf = session.get("classifier")
    xdata = session.get("xtest")
    user_args = session.get("user_args", {})

    if not mmm_classifier:
        return jsonify(
            {"success": False, "error": "No trained model found! Run training first."}
        )

    # Validate save path
    if not save_path or not os.path.isdir(save_path):
        return jsonify(
            {
                "success": False,
                "error": "Invalid directory. Please select a valid folder.",
            }
        )

    # Call deploy() to save the model in the user-specified directory
    try:
        user_args["save_path"] = save_path  # Update args with the user-selected path
        convert_to_onnx(mmm_classifier, save_path, xdata, clf)
        return jsonify(
            {
                "success": True,
                "message": f"Model saved in {save_path}",
                "chat_history": [
                    {
                        "sender": "bot",
                        "text": f"Model successfully saved in {save_path}. What would you like to do next?",
                        "options": [
                            {"value": "reset", "text": "Start Over"},
                            {"value": "exit", "text": "Exit"},
                        ],
                    }
                ],
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
