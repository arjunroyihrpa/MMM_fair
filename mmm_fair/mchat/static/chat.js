document.addEventListener("DOMContentLoaded", function(){
    
    const chatBox = document.getElementById("chat-box");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const resetBtn = document.getElementById("reset-btn");
    const vizcont = document.getElementById("visualization-container");
    const plot3dBox = document.getElementById("plot3d-box");
    const plot2dBox = document.getElementById("plot2d-box");
    const savePathInput = document.getElementById("save-path");
    const saveBtn = document.getElementById("save-btn");

    // === THEME TOGGLE FUNCTIONALITY ===
    const themeToggle = document.getElementById("theme-toggle");
    const body = document.body;

    const savedTheme = localStorage.getItem("mmmfair_theme") || "dark";
    body.classList.add(savedTheme);
    themeToggle.textContent = savedTheme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode";

    themeToggle.addEventListener("click", () => {
        const isDark = body.classList.toggle("dark");
        body.classList.toggle("light", !isDark);
        themeToggle.textContent = isDark ? "☀️ Light Mode" : "🌙 Dark Mode";
        localStorage.setItem("mmmfair_theme", isDark ? "dark" : "light");
    });
    document.addEventListener("DOMContentLoaded", function () {
      
    
      // Set default theme if none stored
      const savedTheme = localStorage.getItem("mmmfair_theme") || "dark";
      body.classList.add(savedTheme);
      themeToggle.textContent = savedTheme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode";
    
      // Add event listener
      themeToggle.addEventListener("click", () => {
        const isDark = body.classList.toggle("dark");
        body.classList.toggle("light", !isDark);
        themeToggle.textContent = isDark ? "☀️ Light Mode" : "🌙 Dark Mode";
        localStorage.setItem("mmmfair_theme", isDark ? "dark" : "light");
      });
    
      // Scroll iframe content into view when loaded
        const observeIframeLoads = () => {
        const iframes = document.querySelectorAll("iframe");
        iframes.forEach((iframe) => {
          iframe.addEventListener("load", () => {
            const chatBox = document.getElementById("chat-box");
            if (chatBox) {
              chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: "smooth" });
            }
          });
        });
      };
    
      setInterval(observeIframeLoads, 2000);
       // Re-check iframes periodically
    });
    // theme logic ends
    // const thetaInput = document.getElementById("theta-input");
    // const thetaBtn = document.getElementById("theta-btn");

    // Global array to track selected features
    window.selectedFeatures = [];

    // Function to render messages in the chatbox with animations
    function renderMessage(sender, text, options = null, isTyping = false, isMarkdown=false) {
        let messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender); // Adds "user" or "bot" class

        let messageWrapper = document.createElement("div");
        messageWrapper.classList.add("message-wrapper");
        
        // Add the text content
        let textDiv = document.createElement("div");
        textDiv.classList.add("message-text");
        messageWrapper.appendChild(messageDiv);
        messageDiv.appendChild(textDiv);
        
        chatBox.appendChild(messageWrapper);
        chatBox.scrollTop = chatBox.scrollHeight;  // Auto-scroll

        if (isTyping && sender === "bot") {
            // If it's bot message and typing animation is requested,
            // add the text with animation, then add buttons or feature selector after text completes
            typeText(textDiv, text, options, messageDiv);
        } else {
            // For user messages or when no typing animation is needed
             if (isMarkdown) {
            // If we want to render Markdown
            textDiv.innerHTML = DOMPurify.sanitize(marked.parse(text));  
            } else {
                textDiv.innerHTML = text.replace(/\n/g, "<br>");
            }
            
            // Add buttons or feature selector immediately for non-animated messages
            if (options) {
                // Check if this is a feature selector type of options
                if (options.type === "features_selector") {
                    // Create and append the feature selector UI
                    const featureSelector = createFeatureSelector(options.available, options.recommended, options.selector_title, options.item_label, options.arg_name);
                    messageDiv.appendChild(featureSelector);
                    // Scroll to the bottom after feature selector is added
                    scrollToBottom();
                } else if (options.length > 0) {
                    // Normal buttons
                    addOptionButtons(messageDiv, options);
                    // Scroll to the bottom after buttons are added
                    scrollToBottom();
                }
            }
        }
    }

    // Dedicated function for scrolling to bottom with smooth animation
    function scrollToBottom() {
        setTimeout(() => {
            chatBox.scrollTo({ 
                top: chatBox.scrollHeight, 
                behavior: "smooth" 
            });
        }, 100); // Short delay to ensure content is rendered
    }

    function scrollResultsToBottom() {
        setTimeout(() => {
            vizcont.scrollTo({ 
                top: vizcont.scrollHeight, 
                behavior: "smooth" 
            });
        }, 100); // Short delay to ensure content is rendered
    }

    // Modified typing effect function to handle feature selector UI
    function typeText(element, text, options, messageDiv, speed = 5) {
        let index = 0;
        let htmlText = text.replace(/\n/g, '<br>');
        element.innerHTML = '';
        
        function type() {
            if (index < htmlText.length) {
                if (htmlText.substr(index, 4) === '<br>') {
                    element.innerHTML += '<br>';
                    index += 4;
                } else {
                    element.innerHTML += htmlText.charAt(index);
                    index++;
                }
                
                // Periodically scroll during typing for longer messages
                if (index % 50 === 0) {
                    chatBox.scrollTop = chatBox.scrollHeight;
                }
                
                setTimeout(type, speed);
            } else {
                // Text animation complete, now add buttons or feature selector if available
                if (options) {
                    // Check if this is a feature selector
                    if (options.type === "features_selector") {
                        // Add a small delay before showing the feature selector
                        setTimeout(() => {
                            const featureSelector = createFeatureSelector(options.available, options.recommended, options.selector_title, options.item_label, options.arg_name);
                            messageDiv.appendChild(featureSelector);
                            // Scroll to bottom after feature selector is added
                            scrollToBottom();
                        }, 300);
                    } else if (options.length > 0) {
                        // Normal buttons
                        setTimeout(() => {
                            addOptionButtons(messageDiv, options);
                            // Scroll to bottom after buttons are added
                            scrollToBottom();
                        }, 300);
                    }
                }
            }
        }
        type();
    }

    // Add option buttons to a message
    function addOptionButtons(messageDiv, options) {
        // Check if buttons already exist
        if (messageDiv.querySelector('.option-buttons')) {
            return;
        }
        
        let buttonsDiv = document.createElement("div");
        buttonsDiv.classList.add("option-buttons");
        
        options.forEach(option => {
            let button = document.createElement("button");
            button.classList.add("option-button");
            button.textContent = option.text;
            button.dataset.value = option.value;
            
            // Handle button clicks
            button.addEventListener("click", function() {
                handleOptionButtonClick(option.value, option.text);
            });
            
            buttonsDiv.appendChild(button);
        });
        
        // Add buttons with a fade-in effect
        buttonsDiv.style.opacity = '0';
        messageDiv.appendChild(buttonsDiv);
        
        // Trigger animation and scroll
        setTimeout(() => {
            buttonsDiv.style.transition = 'opacity 0.5s ease';
            buttonsDiv.style.opacity = '1';
            scrollToBottom();
        }, 50);
    }

    // Create and display the feature selection UI
    window.createFeatureSelector = function(availableFeatures, recommendedFeatures, title, itemLabel, argName = 'prots') {
        // Clear any previous selection
        window.selectedFeatures = [];
        
        // Create the feature selector container
        const selectorContainer = document.createElement('div');
        selectorContainer.classList.add('feature-selector-container');
        
        // Create header with instructions
        const header = document.createElement('h4');
        header.textContent = `${title}`;
        selectorContainer.appendChild(header);
        
        // Create the recommended features section if available
        if (recommendedFeatures && recommendedFeatures.length > 0) {
            const recommendedSection = document.createElement('div');
            recommendedSection.classList.add('recommended-features');
            
            const recommendedLabel = document.createElement('div');
            recommendedLabel.classList.add('section-label');
            recommendedLabel.textContent = 'Recommended:';
            recommendedSection.appendChild(recommendedLabel);
            
            const recommendedButtons = document.createElement('div');
            recommendedButtons.classList.add('recommended-buttons');
            
            recommendedFeatures.forEach(feature => {
                const button = document.createElement('button');
                button.classList.add('feature-button', 'recommended');
                button.textContent = feature;
                button.dataset.feature = feature;
                
                button.addEventListener('click', function() {
                    selectFeature(feature, button);
                    // Scroll to bottom when a feature is selected, as tags will appear
                    scrollToBottom();
                });
                
                recommendedButtons.appendChild(button);
            });
            
            recommendedSection.appendChild(recommendedButtons);
            selectorContainer.appendChild(recommendedSection);
        }
        
        // Create all available features section
        if (availableFeatures && availableFeatures.length > 0) {
            const allFeaturesSection = document.createElement('div');
            allFeaturesSection.classList.add('all-features');
            
            const allFeaturesLabel = document.createElement('div');
            allFeaturesLabel.classList.add('section-label');

            allFeaturesLabel.textContent = `All Available ${itemLabel}:`;
            allFeaturesSection.appendChild(allFeaturesLabel);
            
            // Create dropdown for all features
            const selectBox = document.createElement('select');
            selectBox.id = 'feature-dropdown';
            selectBox.classList.add('feature-dropdown');
            
            // Add placeholder option
            const placeholder = document.createElement('option');
            placeholder.value = '';
            placeholder.textContent = `-- Select a ${itemLabel} --:`;
            placeholder.disabled = true;
            placeholder.selected = true;
            selectBox.appendChild(placeholder);
            
            // Add all features to dropdown
            availableFeatures.forEach(feature => {
                const option = document.createElement('option');
                option.value = feature;
                option.textContent = feature;
                selectBox.appendChild(option);
            });
            
            allFeaturesSection.appendChild(selectBox);
            
            // Add button to add selected feature
            const addButton = document.createElement('button');
            addButton.classList.add('add-feature-btn');
            addButton.textContent = `Add ${itemLabel}`;
            addButton.addEventListener('click', function() {
                const selectedFeature = selectBox.value;
                if (selectedFeature) {
                    selectFeature(selectedFeature);
                    selectBox.value = ''; // Reset dropdown
                    // Scroll to bottom when a feature is added
                    scrollToBottom();
                }
            });
            
            allFeaturesSection.appendChild(addButton);
            selectorContainer.appendChild(allFeaturesSection);
        }
        
        // Create selected features section
        const selectedSection = document.createElement('div');
        selectedSection.classList.add('selected-features');
        
        const selectedLabel = document.createElement('div');
        selectedLabel.classList.add('section-label');
        selectedLabel.textContent = 'Your Selection:';
        selectedSection.appendChild(selectedLabel);
        
        // Container for selected feature tags
        const selectedTags = document.createElement('div');
        selectedTags.classList.add('selected-tags');
        selectedTags.id = 'selected-tags';
        selectedSection.appendChild(selectedTags);
        
        selectorContainer.appendChild(selectedSection);
        
        // Add submit button
        const submitBtn = document.createElement('button');
        submitBtn.classList.add('submit-features-btn');
        submitBtn.textContent = 'Submit Selection';

        submitBtn.addEventListener('click', function() {
            submitFeatureSelection(argName);
        });

        selectorContainer.appendChild(submitBtn);
        
        // Ensure the container is fully visible
        setTimeout(scrollToBottom, 100);
        
        return selectorContainer;
    }

    // Function to select a feature
    window.selectFeature = function(feature, button = null) {
        // Check if already selected
        if (window.selectedFeatures.includes(feature)) {
            return;
        }
        
        // Add to selected features array
        window.selectedFeatures.push(feature);
        
        // Highlight the button if provided
        if (button) {
            button.classList.add('selected');
        }
        
        // Add tag to selected tags
        const selectedTags = document.getElementById('selected-tags');
        if (!selectedTags) return;
        
        const tag = document.createElement('div');
        tag.classList.add('feature-tag');
        
        const tagText = document.createElement('span');
        tagText.textContent = feature;
        tag.appendChild(tagText);
        
        const removeBtn = document.createElement('button');
        removeBtn.classList.add('remove-tag');
        removeBtn.innerHTML = '&times;';
        removeBtn.addEventListener('click', function() {
            removeFeature(feature);
            tag.remove();
            
            // Remove selected class from button if exists
            if (button) {
                button.classList.remove('selected');
            }
        });
        
        tag.appendChild(removeBtn);
        selectedTags.appendChild(tag);
        
        // Scroll to see the newly added tag
        scrollToBottom();
        
        // Send to server
        fetch('/select_feature', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                feature: feature,
                selector_type: selectorType
            })
        })
        .then(response => response.json())
        .catch(error => console.error('Error selecting feature:', error));
    }

    // Function to remove a feature
    window.removeFeature = function(feature) {
        // Remove from array
        const index = window.selectedFeatures.indexOf(feature);
        if (index > -1) {
            window.selectedFeatures.splice(index, 1);
        }
        
        // Find and unhighlight any buttons for this feature
        document.querySelectorAll('.feature-button').forEach(button => {
            if (button.dataset.feature === feature) {
                button.classList.remove('selected');
            }
        });
    }

    // Function to submit feature selection
    window.submitFeatureSelection = function(selectorType = 'prots') {

        console.log('DEBUG submitFeatureSelection called with selectorType:', selectorType);
        console.log('DEBUG selectedFeatures:', window.selectedFeatures);
        if (window.selectedFeatures.length === 0 && selectorType === 'prots') {
            alert('Please select at least one protected attribute.');
            return;
        }
        showLoading();
        
        const buttonValue = selectorType === 'baseline_models' ? 'submit_baseline_models' : 'submit_features';
        
        console.log('DEBUG buttonValue will be:', buttonValue);

        fetch('/ask_chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: '',
                button_value: buttonValue,
                selected_features: window.selectedFeatures
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            // Clear the feature selector UI
            const messageDiv = document.querySelector('.feature-selector-container');
            if (messageDiv) {
                const parentMessage = messageDiv.closest('.message');
                if (parentMessage) {
                    // Keep just the text part and remove the selector
                    const textContent = parentMessage.querySelector('.message-text');
                    if (textContent) {
                        // textContent.innerHTML = `Selected protected attributes: ${window.selectedFeatures.join(', ')}`;
                        messageDiv.remove();
                    }
                }
            }
            
            // Display subsequent messages
            if (data.chat_history && data.chat_history.length > 0) {
                data.chat_history.forEach(chat => {
                    renderMessage(chat.sender, chat.text, chat.options, true);
                });
            }
            
            // Auto-scroll to new messages
            scrollToBottom();
        })
        .catch(error => {
            console.error('Error submitting features:', error);
            alert('An error occurred while submitting features.');
        });
    }

    // Handle option button clicks
    function handleOptionButtonClick(value, text) {
        // First, display the user's choice
        renderMessage("user", text);
        
        // If it's a provider selection (for LLM), treat it specially
        const llmProviders = ['openai', 'chatgpt', 'groq', 'groqai', 'together', 'togetherai'];
        if (llmProviders.includes(value.toLowerCase())) {
            showLoading(); 
            fetch("/ask_chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({message: value})
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.require_api_key) {
                    // Same API key input UI creation as in sendMessage
                    const providerName = data.provider || value;
                    const botMessage = data.message || `To use AI explanation features, please enter your ${providerName} API key:`;
                    
                    // Create message with the API key input UI
                    const messageDiv = document.createElement("div");
                    messageDiv.classList.add("message", "bot");
                    
                    const messageWrapper = document.createElement("div");
                    messageWrapper.classList.add("message-wrapper");
                    
                    const textDiv = document.createElement("div");
                    textDiv.classList.add("message-text");
                    textDiv.innerHTML = botMessage.replace(/\n/g, '<br>');
                    
                    messageDiv.appendChild(textDiv);
                    messageWrapper.appendChild(messageDiv);
                    chatBox.appendChild(messageWrapper);
                    
                    // Add the API key input UI
                    const apiKeyInput = createAPIKeyInput(providerName);
                    messageDiv.appendChild(apiKeyInput);
                    
                    // Focus on the input field
                    setTimeout(() => {
                        const input = document.getElementById('api-key-input');
                        if (input) input.focus();
                    }, 100);
                    
                    // Scroll to bottom after adding the input UI
                    scrollToBottom();
                    return;
                }
                
                // Handle regular responses
                if (data.chat_history && data.chat_history.length > 0) {
                    data.chat_history.forEach(chat => {
                        renderMessage(chat.sender, chat.text, chat.options, true);
                    });
                }
                
                // Scroll to bottom
                scrollToBottom();
            })
            .catch(error => console.error("Error:", error));
        } else if (value.startsWith("nprotg_")) {
            // If it's a non-protected group selection
            const parts = value.split("_");
            // Extract the actual value (skip "nprotg" and the attribute name)
            const actualValue = parts.slice(2).join("_");
            showLoading(); 
            fetch("/ask_chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    message: "",
                    button_value: value,
                    nprotgs_value: actualValue
                })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                // Display responses
                if (data.chat_history && data.chat_history.length > 0) {
                    data.chat_history.forEach(chat => {
                        renderMessage(chat.sender, chat.text, chat.options, true);
                    });
                }
                
                // Auto-scroll to new messages
                scrollToBottom();
            })
            .catch(error => console.error("Error:", error));
        } else if (value === "upload_data") {
            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.accept = ".csv";
            fileInput.style.display = "none";
            document.body.appendChild(fileInput);
            fileInput.addEventListener("change", function () {
            const file = fileInput.files[0];
            if (file) {
                const formData = new FormData();
                formData.append("file", file);
                formData.append("button_value", "upload_data");
                formData.append("message", "upload_data");
                showLoading();

                fetch("/ask_chat", {
                method: "POST",
                body: formData,
                })
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    if (data.chat_history?.length > 0) {
                        data.chat_history.forEach(chat => {
                        renderMessage(chat.sender, chat.text, chat.options, true);
                    });
                        } 
                    else {
                        renderMessage("bot", `❌ Upload failed: ${data.error}`);
                        }
                 scrollToBottom();   
                })
                .catch(error => {
                    hideLoading();
                    console.error("Upload error:", error);
                    renderMessage("bot", "❌ An error occurred during upload.");
                });
                }
                document.body.removeChild(fileInput);
            });
            fileInput.click(); 
        }
        
        
        else {
            sendButtonChoice(value);
        }
    }

    function updateModelWithTheta(thetaValue) {
        fetch("/update_model", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({button_value: thetaValue})
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Display success message
                if (data.message) {
                    renderMessage("bot", data.message);
                }
                
                // Update the fairness report visualization
                if (data.plot_fair_url) {
                    console.log("DEBUG: Updating plot2dBox with new report");
                    plot2dBox.innerHTML = `<iframe src="${data.plot_fair_url}?t=${Date.now()}" width="100%" height="400px" frameborder="0"></iframe>`;
                }
                
                // Display any additional chat messages
                if (data.chat_history && data.chat_history.length > 0) {
                    data.chat_history.forEach(chat => {
                        renderMessage(chat.sender, chat.text, chat.options, true);
                    });
                }
                
                // Scroll after all updates are done
                scrollToBottom();
            } else {
                renderMessage("bot", `Error updating model: ${data.error}`);
                scrollToBottom();
            }
        })
        .catch(error => {
            console.error("Error updating model:", error);
            renderMessage("bot", "An error occurred while updating the model. Please try again.");
            scrollToBottom();
        });
    }
    
    // Send button choice to server
    function sendButtonChoice(buttonValue) {
        // Handle visualization request buttons
        if (buttonValue === "visualize_yes" || buttonValue === "visualize_no") {
            showLoading(); 
            fetch("/ask_chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    message: "",
                    button_value: buttonValue
                })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                // Handle redirect to visualization endpoint
                if (data.redirect) {
                    hideLoading();
                    handleVisualizationRedirect(data.redirect);
                    return;
                }
                
                // Regular flow for chat messages
                if (data.chat_history && data.chat_history.length > 0) {
                    data.chat_history.forEach(chat => {
                        renderMessage(chat.sender, chat.text, chat.options, true);
                    });
                }
                
                // Auto-scroll after content is loaded
                scrollToBottom();
            })
            .catch(error => {
                console.error("Error:", error);
                renderMessage("bot", "An error occurred. Please try again.");
                scrollToBottom();
            });
        } else {
            // Original handling for other button types
            showLoading(); 
            fetch("/ask_chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    message: "",
                    button_value: buttonValue
                })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.chat_history && data.chat_history.length > 0) {
                    data.chat_history.forEach(chat => {
                        renderMessage(chat.sender, chat.text, chat.options, true);
                    });
                }

                console.log("DEBUG: plots array: ", data);

                // Load Plot HTML Files Dynamically
                if (data.plots && Array.isArray(data.plots)) {
                    data.plots.forEach(plot => {
                        displayDataVisualization(plot);
                    });
                }

                if (data.html_divs && Array.isArray(data.html_divs)) {
                    data.html_divs.forEach(html_div => {
                        displayHtmlDiv(html_div);
                    });
                }

                // if (data.plot_all_url) {
                //     plot3dBox.innerHTML = `<iframe src="${data.plot_all_url}" width="100%" height="400px" frameborder="0"></iframe>`;
                // }
                // if (data.plot_fair_url) {
                //     plot2dBox.innerHTML = `<iframe src="${data.plot_fair_url}" width="100%" height="400px" frameborder="0"></iframe>`;
                // }
    
                // Auto-scroll after all content is loaded
                scrollToBottom();
            })
            .catch(error => {
                console.error("Error:", error);
                renderMessage("bot", "An error occurred. Please try again.");
                scrollToBottom();
            });
        }
    }
    // Show a welcome message with typing animation
    function showWelcomeMessage() {
        const welcomeOptions = [
            // {value: "MMM_Fair", text: "MMM_Fair (AdaBoost)"},
            // {value: "MMM_Fair_GBT", text: "MMM_Fair_GBT (Gradient Boosting)"},
            // {value: "default", text: "Run with default parameters"}
            {value: "Adult", text: "🧑 Adult (UCI)"},
            {value: "Bank", text: "🏦 Bank Marketing (UCI)"},
            {value: "Credit", text: "💳 Credit Default (UCI)"},
            {value: "kdd", text: "🧑 Census-Income KDD (UCI)"},
            {value: "upload_data", text: "📁 Upload your own Data (currently supported types: '.csv'"},
            {value: "default", text: "Run with default setup on Adult data"}
        ];
        
        renderMessage("bot", 
            "👋 Hello! Welcome to MMM-Fair Chat.\n\n" +
            "Let's get started by selecting a dataset to work with.\n You can choose from well-known public datasets or upload your own CSV file.\n" +
            "Please select an option below:", 
            welcomeOptions, 
            true
        );
    }

    function loadSessionOnLoad() {
        fetch("/get_session_state")
        .then(response => response.json())
        .then(data => {

            if (data.chat_history && data.chat_history.length > 0) {
                data.chat_history.forEach(msg => {
                    renderMessage(msg.sender, msg.text, msg.options, false);
                });
            } else {
                showWelcomeMessage();
            }
            console.log("DEBUG: plots array: ", data);

            if (data.plots && Array.isArray(data.plots)) {
                data.plots.forEach(plot => {
                    displayDataVisualization(plot);
                });
            }

            if (data.html_divs && Array.isArray(data.html_divs)) {
                data.html_divs.forEach(html_div => {
                    displayHtmlDiv(html_div);
                });
            }

            // if (data.plot_all_url) {
            //     plot3dBox.innerHTML = `<iframe src="${data.plot_all_url}" width="100%" height="400px" frameborder="0"></iframe>`;
            // }
            // if (data.plot_fair_url) {
            //     plot2dBox.innerHTML = `<iframe src="${data.plot_fair_url}" width="100%" height="400px" frameborder="0"></iframe>`;
            // }
            
            // Scroll to bottom after loading all content
            scrollToBottom();
        })
        .catch(error => {
            console.error("Error loading session state:", error);
            showWelcomeMessage();
        });
    }

    // Reset chat history
    function resetChat() {
        fetch("/reset_chat")
        .then(() => {
            Array.from(vizcont.children).forEach(child => {
                vizcont.removeChild(child);
            });
            Array.from(chatBox.children).forEach(child => {
                chatBox.removeChild(child);
            });
            showWelcomeMessage();
        })
        .catch(error => console.error("Error resetting chat:", error));
    }

    function handleVisualizationRedirect(redirectPath) {
        fetch(redirectPath, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.success && data.plots) {
                // Display the plot using the HTML directly
                if (data.plots && Array.isArray(data.plots)) {
                    data.plots.forEach(plot => {
                        displayDataVisualization(plot);
                    });
                }
                
                // Display any additional chat messages
                if (data.chat_history && data.chat_history.length > 0) {
                    data.chat_history.forEach(chat => {
                        renderMessage(chat.sender, chat.text, chat.options, true);
                    });
                }
            } else if (data.error) {
                renderMessage("bot", `Error: ${data.error}`, null, true);
            }
            
            // Scroll after everything is loaded
            scrollToBottom();
        })
        .catch(error => {
            console.error('Error:', error);
            renderMessage("bot", "Sorry, there was an error generating the visualization.", null, true);
            scrollToBottom();
        });
    }
    

    function updateDataVisualization(data_to_update) {

        const existing_iframe = document.getElementById(data_to_update.existing_id);
        
        const args = data_to_update.updated_data
        

        // Apply args to iframe 
        for (const [key, value] of Object.entries(args)) {
            if (key === 'style' && typeof value === 'object') {
                // Apply styles
                for (const [styleKey, styleValue] of Object.entries(value)) {
                    existing_iframe.style[styleKey] = styleValue;
                }
            } else {
                // Apply as attribute
                existing_iframe.setAttribute(key, value);
            }
        }

        // document.getElementById('visualization-container').style.display = 'block';
        // scrollResultsToBottom();
    }

    function displayDataVisualization(plot_data) {
        args = plot_data
        const iframe = document.createElement('iframe');
        // iframe.classList.add('visualization-iframe');
        // iframe.style.width = '100%';
        // iframe.style.height = '100px';
        // iframe.style.border = 'none';
        // iframe.style.overflow = 'hidden';
    
        vizcont.appendChild(iframe);
  
        // iframe.srcdoc = plotHtml;

        // iframe.style.width = '100%';
        // iframe.style.height = '500px';
        // iframe.style.overflow = 'hidden';
        // iframe.style.border = 'None';

        // Apply args to iframe 
        for (const [key, value] of Object.entries(args)) {
            if (key === 'style' && typeof value === 'object') {
                // Apply styles
                for (const [styleKey, styleValue] of Object.entries(value)) {
                    iframe.style[styleKey] = styleValue;
                }
            } else {
                // Apply as attribute
                iframe.setAttribute(key, value);
            }
        }

        document.getElementById('visualization-container').style.display = 'block';

        scrollResultsToBottom();
    }

    function displayHtmlDiv(htmlContent) {
        const div = document.createElement('div');
        div.innerHTML = htmlContent;
        div.style.width = '100%';
        div.style.overflow = 'hidden';
    
        vizcont.appendChild(div);
    
        document.getElementById('visualization-container').style.display = 'block';
    
        scrollResultsToBottom();
    }
    function showLoading() {
          document.getElementById("loading-overlay").style.display = "flex";
        }
        
        function hideLoading() {
          document.getElementById("loading-overlay").style.display = "none";
        }


    // Send message function
    function sendMessage() {
        let userMessage = chatInput.value.trim();
        if (!userMessage) return;
        chatInput.value = "";
    
        renderMessage("user", userMessage);
        showLoading(); 
    
        fetch("/ask_chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({message: userMessage})
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();

            if (data.require_api_key) {
                const providerName = data.provider || "the selected LLM";
                const botMessage = data.message || `To use AI explanation features, please enter your ${providerName} API key:`;
                
                // Create a message with the API key input UI
                const messageDiv = document.createElement("div");
                messageDiv.classList.add("message", "bot");
                
                const messageWrapper = document.createElement("div");
                messageWrapper.classList.add("message-wrapper");
                
                const textDiv = document.createElement("div");
                textDiv.classList.add("message-text");
                textDiv.innerHTML = botMessage.replace(/\n/g, '<br>');
                
                messageDiv.appendChild(textDiv);
                messageWrapper.appendChild(messageDiv);
                chatBox.appendChild(messageWrapper);
                
                // Add the API key input UI
                const apiKeyInput = createAPIKeyInput(providerName);
                messageDiv.appendChild(apiKeyInput);
                
                // Focus on the input field
                setTimeout(() => {
                    const input = document.getElementById('api-key-input');
                    if (input) input.focus();
                }, 100);
                
                // Scroll to bottom after adding the input UI
                scrollToBottom();
                return;
            }

            // Handle redirect to visualization endpoint
            if (data.redirect) {
                hideLoading();
                handleVisualizationRedirect(data.redirect);
                return;
            }
            
            if (data.chat_history && data.chat_history.length > 0) {
                data.chat_history.forEach(chat => {
                    renderMessage(chat.sender, chat.text, chat.options, true);
                });
            }
            console.log("DEBUG: plots array: ", data);
            
            if (data.plots && Array.isArray(data.plots)) {
                data.plots.forEach(plot => {
                    displayDataVisualization(plot);
                });
            }

            if (data.html_divs && Array.isArray(data.html_divs)) {
                data.html_divs.forEach(html_div => {
                    displayHtmlDiv(html_div);
                });
            }

            
            
            // if (data.plot_all_url) {
            //     plot3dBox.innerHTML = `<iframe src="${data.plot_all_url}" width="100%" height="400px" frameborder="0"></iframe>`;
            // }
            // if (data.plot_fair_url) {
            //     plot2dBox.innerHTML = `<iframe src="${data.plot_fair_url}" width="100%" height="400px" frameborder="0"></iframe>`;
            // }
    
            // Auto-scroll after everything is loaded
            scrollToBottom();
        })
        .catch(error => {
                    hideLoading(); 
                    console.error("Error:", error);
            });
    }
    
    function createAPIKeyInput(providerName) {
        const apiKeyContainer = document.createElement('div');
        apiKeyContainer.classList.add('api-key-container');
        
        const header = document.createElement('h4');
        header.textContent = `Enter your ${providerName.toUpperCase()} API Key`;
        apiKeyContainer.appendChild(header);
        
        const inputGroup = document.createElement('div');
        inputGroup.classList.add('api-key-input-group');
        
        const keyInput = document.createElement('input');
        keyInput.type = 'password';
        keyInput.id = 'api-key-input';
        keyInput.placeholder = 'Paste your API key here';
        keyInput.classList.add('api-key-input');
        inputGroup.appendChild(keyInput);
        
        const submitBtn = document.createElement('button');
        submitBtn.classList.add('api-key-submit-btn');
        submitBtn.textContent = 'Submit';
        submitBtn.addEventListener('click', function() {
            submitAPIKey(keyInput.value, providerName);
        });
        
        // Also allow Enter key to submit
        keyInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                submitAPIKey(keyInput.value, providerName);
            }
        });
        
        inputGroup.appendChild(submitBtn);
        apiKeyContainer.appendChild(inputGroup);
        
        // Add help text
        const helpText = document.createElement('p');
        helpText.classList.add('api-key-help');
        helpText.innerHTML = 'Your API key is needed to generate explanations. It will be used only for the current session.';
        apiKeyContainer.appendChild(helpText);
        
        return apiKeyContainer;
    }

    // Function to submit the API key
    function submitAPIKey(apiKey, providerName) {
        if (!apiKey) {
            alert('Please enter a valid API key');
            return;
        }
        
        fetch("/provide_api_key", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                api_key: apiKey,
                model: providerName
            })
        })
        .then(response => response.json())
        .then(data => {
            // Find and remove the API key input UI
            const apiKeyContainer = document.querySelector('.api-key-container');
            if (apiKeyContainer) {
                const parentMessage = apiKeyContainer.closest('.message');
                if (parentMessage) {
                    const textContent = parentMessage.querySelector('.message-text');
                    if (textContent) {
                        // Update the message to show success/error
                        textContent.innerHTML = data.success 
                            ? data.message 
                            : `Error: ${data.error}`;
                        apiKeyContainer.remove();
                    }
                }
            }
            
            if (data.success) {
                // Auto-trigger the explanation
                renderMessage('bot', 'Generating explanation, please wait...', null, true);
                showLoading(); 
                // Send a request to generate the explanation
                fetch("/ask_chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({message: "explain"})
                })
                .then(response => response.json())
                .then(response => {
                    hideLoading();
                    // Handle the explanation response
                    if (response.chat_history && response.chat_history.length > 0) {
                        response.chat_history.forEach(chat => {
                            renderMessage(chat.sender, chat.text, chat.options, true);
                        });
                    }
                    
                    // Scroll to bottom after everything is loaded
                    scrollToBottom();
                })
                .catch(error => {
                    hideLoading();
                    console.error("Error generating explanation:", error);
                    renderMessage("bot", "An error occurred while generating the explanation.", null, true);
                    scrollToBottom();
                });
            } else {
                // If there was an error, allow the user to try again
                renderMessage("bot", "Would you like to try again with a different API key?", [
                    {"value": "retry_api_key", "text": "Try again"},
                    {"value": "skip_explanation", "text": "Skip explanation"}
                ], true);
                scrollToBottom();
            }
        })
        .catch(error => {
            console.error("Error submitting API key:", error);
            renderMessage("bot", "An error occurred while submitting your API key. Please try again.", null, true);
            scrollToBottom();
        });
    }
    
    
    // vizcont.addEventListener("click", function(e) {
    //     // Handle Theta Selection and Model Update
    //     if (e.target && e.target.id === "theta-btn") {
    //         const thetaInput = document.getElementById("theta-input"); // Get it fresh at click time

    //         const thetaValue = thetaInput.value.trim();
    //         if (!thetaValue) {
    //             alert("Please enter a valid Theta index!");
    //             return;
    //         }
    
    //         fetch("/update_model", {
    //             method: "POST",
    //             headers: { "Content-Type": "application/json" },
    //             body: JSON.stringify({ theta: thetaValue })
    //         })
    //         .then(response => response.json())
    //         .then(data => {
    //             if (data.success) {

    //                 if (data.update_plots && Array.isArray(data.update_plots)) {
    //                     data.update_plots.forEach(update_plot => {
    //                         updateDataVisualization(update_plot);
    //                     });
    //                 }

    //                 alert(`Model updated successfully with Theta ${thetaValue}`);
    
    //                 if (data.plot_fair_url) {
    //                     console.log("DEBUG: Updating plot2dBox with new report");
    //                     plot2dBox.innerHTML = `<iframe src="${data.plot_fair_url}?t=${Date.now()}" width="100%" height="400px" frameborder="0"></iframe>`;
    //                 }
    //             } else {
    //                 alert(`Error updating model: ${data.error}`);
    //             }
    //         })
    //         .catch(error => console.error("Error updating model:", error));
    //     }
    //     // Handle save model button
    // if (e.target && e.target.id === "save-btn") {
    //     saveModel();
    // }
    // });  
    document.addEventListener("click", function(e) {
      if (e.target && e.target.id === "theta-btn") {
        const thetaInput = document.getElementById("theta-input");
        const thetaValue = thetaInput.value.trim();
        if (!thetaValue) {
          alert("Please enter a valid Theta index!");
          return;
        }
    
        fetch("/update_model", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ theta: thetaValue })
        })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            alert(`Model updated successfully with Theta ${thetaValue}`);
            if (data.plots && Array.isArray(data.plots)) {
                data.plots.forEach(plot => {
                    updateDataVisualization(plot);
                });
                }
          } else {
            alert(`Error updating model: ${data.error}`);
          }
        })
        .catch(error => console.error("Error updating model:", error));
      }
    
      if (e.target && e.target.id === "save-btn") {
        saveModel();  
      }
    });

    // Handle Save Model Button
    function saveModel() {
        //let savePath = savePathInput.value.trim();
        const savePathInput = document.getElementById("save-path");
        const savePath = savePathInput.value.trim();
        if (!savePath) {
            renderMessage("bot", "Please enter a valid directory path to save the model!");
            return;
        }
    
        fetch("/save_model", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({save_path: savePath})
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                renderMessage("bot", `Model saved successfully in: ${savePath}`);
                
                // Display any additional chat messages
                if (data.chat_history && data.chat_history.length > 0) {
                    data.chat_history.forEach(chat => {
                        renderMessage(chat.sender, chat.text, chat.options);
                    });
                }
            } else {
                renderMessage("bot", `Error saving model: ${data.error}`);
            }
            
            // Scroll after model save response
            scrollToBottom();
        })
        .catch(error => {
            console.error("Error saving model:", error);
            renderMessage("bot", "An error occurred while saving the model. Please try again.");
            scrollToBottom();
        });
    }

    // Additional event listener for iframe load to ensure scrolling
    // This helps when iframes are loaded and change the document height
    const observeIframeLoads = () => {
        const iframes = document.querySelectorAll('iframe');
        iframes.forEach(iframe => {
            iframe.addEventListener('load', () => {
                scrollToBottom();
            });
        });
    };

    // Periodically check for new iframes and add listeners
    setInterval(observeIframeLoads, 2000);

    // Event Listeners
    if (sendBtn) sendBtn.addEventListener("click", sendMessage);
    if (chatInput) chatInput.addEventListener("keydown", function(e){
        if (e.key === "Enter") sendMessage();
    });
    if (resetBtn) resetBtn.addEventListener("click", resetChat);
    if (saveBtn) saveBtn.addEventListener("click", saveModel);

    // Initialize
    loadSessionOnLoad();

    
});


