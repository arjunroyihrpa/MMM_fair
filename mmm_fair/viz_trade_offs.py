import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import textwrap


def plot2d(
    x,
    y,
    z=None,
    axis_names=["X", "Y", "Z"],
    title="2D Scatter Plot",
    desc=(
        "🔹 This plot visualizes 3D data projected onto a 2D plane if Z is provided.<br>"
        "🔹 Marker color and size represent the third dimension (Z values).<br>"
        "🔹 If Z is not provided, it simply plots X vs. Y.<br>"
        "🔹 Hover over points to see their coordinates.<br>"
        "<br> _________________________________________________________ <br>"
    ),
    html=False,
):
    """
    Generates a 2D scatter plot.

    Args:
        x (list): X-axis values.
        y (list): Y-axis values.
        z (list or None): Z values (used for color/size if provided).
        axis_names (list): Names for X, Y, and Z axes.
        title (str): Title of the plot.
        desc (str): Description text for annotations.
        html (bool): If True, return the plot as HTML instead of showing it.

    Returns:
        str or None: Returns an HTML string if `html=True`, otherwise displays the plot.
    """

    # Wrap title for better display
    title_wrap = 100
    wrapped_title = "<br>".join(textwrap.wrap(title, width=title_wrap))

    # Check if Z is provided
    use_z = z is not None

    # Create hover text showing full (X, Y, Z) coordinates if Z is provided
    if use_z:
        desc = (
            f"🔹 This plot visualizes {axis_names[0]}, {axis_names[1]}, and {axis_names[2]} data projected onto a 2D plane.<br>"
            f"🔹 Marker color and size represent the third dimension ({{axis_names[2]}} values).<br>"
            "🔹 Hover over points to see their coordinates.<br>"
            "<br> _________________________________________________________ <br>"
        )
        hover_text = [f"({x[i]}, {y[i]}, {z[i]})" for i in range(len(x))]
    else:
        desc = (
            f"🔹 This plot visualizes {axis_names[0]}, and {axis_names[1]} data projected onto a 2D plane.<br>"
            "🔹 Hover over points to see their coordinates.<br>"
            "<br> _________________________________________________________ <br>"
        )
        hover_text = [f"({x[i]}, {y[i]})" for i in range(len(x))]

    # Create figure
    fig = go.Figure()

    # Add 2D scatter plot
    marker_dict = {
        "size": (
            10
            if not use_z
            else [10 + (val - min(z)) / (max(z) - min(z) + 1) * 20 for val in z]
        ),
        "color": "blue" if not use_z else z,  # Default color if Z is not provided
        "colorscale": "Viridis" if use_z else None,
        "showscale": use_z,
        "colorbar": dict(title=f"{axis_names[2]} Values") if use_z else None,
    }

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=marker_dict,
            text=hover_text,
            hoverinfo="text",
            name="Pareto Front" if use_z else "Data Points",
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=wrapped_title,
            font=dict(size=18),
        ),
        xaxis_title=f"X: {axis_names[0]}",
        yaxis_title=f"Y: {axis_names[1]}",
        annotations=[
            go.layout.Annotation(
                showarrow=False,
                text=desc,
                x=1.02,  # Position text to the right
                y=1.02,
                xref="paper",
                yref="paper",
                font=dict(size=14, color="black"),
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="black",
                borderwidth=2,
                borderpad=10,
                align="left",
            )
        ],
        width=300,  # Set plot width
        height=300,  # Set plot height
    )

    #  Return HTML if requested
    if html:
        return fig.to_html(
            full_html=False, include_plotlyjs=True
        )  # Return HTML representation of plot
    else:
        fig.show()  # Display the plot normally


def plot3d(
    x=[1, 2, 3, 4, 5],
    y=[2, 3, 1, 4, 5],
    z=[3, 1, 2, 5, 4],
    theta=[1, 2, 3, 4, 5],
    criteria="all",
    axis_names=["X", "Y", "Z"],
    title="3D Scatter Plot of Pareto front.",
    desc=(
        "🔍Hover over points to see info about associated Theta.<br>"
        "⚖️ Points show different loss trade-offs<br>"
        "🧩 Theta can be used to update the model<br>"
    ),
    html=False,
):
    """
    Generates a 3D scatter plot of Pareto front.

    Args:
        x (list): X-axis values.
        y (list): Y-axis values.
        z (list): Z-axis values.
        theta (list): Theta values.
        criteria (str): Criteria for selection.
        axis_names (list): List of axis names [X, Y, Z].
        title (str): Title of the plot.
        desc (str): Description text to be added.
        html (bool): If True, return the plot as HTML instead of showing it.

    Returns:
        str or None: Returns an HTML string if `html=True`, otherwise displays the plot.
    """

    # Wrap title for better display
    title_wrap = 100
    wrapped_title = "<br>".join(textwrap.wrap(title, width=title_wrap))

    # Hidden data for hover text
    hidden_data = [f"Theta: {theta[i]}" for i in range(len(theta))]

    # Create figure
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])

    # Color values
    color_values = np.array(x) + np.array(y) + np.array(z)
    color_label = "Sum of losses (darker better)"

    # Add 3D scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=8,
                color=color_values,  # Color based on the chosen attribute
                colorscale="Viridis",  # Try 'Plasma', 'Cividis', 'Inferno', etc.
                showscale=True,
                colorbar=dict(title=color_label),
            ),
            name="Points",
            text=hidden_data,
            customdata=hidden_data,
        ),
        row=1,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{criteria}-objectives<br>{wrapped_title}",
            font=dict(size=12),
        ),
        scene=dict(
            xaxis_title=f"X: {axis_names[0]}",
            yaxis_title=f"Y: {axis_names[1]}",
            zaxis_title=f"Z: {axis_names[2]}",
        ),
        annotations=[
            go.layout.Annotation(
                showarrow=False,
                text=desc,
                x=1.02,  # text to the right outside the plot
                y=1.02,
                xref="paper",
                yref="paper",
                font=dict(size=10, color="black"),
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="black",
                borderwidth=2,
                borderpad=10,
                align="left",
            )
        ],
        width=800,  # width of the plot
        height=400,
    )

    # **NEW FEATURE**: Return HTML if requested
    if html:
        return fig.to_html(
            full_html=True, include_plotlyjs=True
        )  # Return HTML representation of plot
    else:
        fig.show()  # Display the plot normally


# import plotly.graph_objects as go
# import numpy as np
# import textwrap
# from plotly.subplots import make_subplots

# import plotly.graph_objects as go
# import numpy as np
# import textwrap
import plotly.express as px


def plot_spider(
    objectives,
    theta=None,
    baseline_results=None,
    criteria="all",
    axis_names=None,
    title="Multi-objective Line Chart",
    desc=(
        "🔍Hover over lines to see info about associated Theta.<br>"
        "⚖️ Lines show different loss trade-offs<br>"
        "🧩 Theta can be used to update the model<br>"
    ),
    html=False,
):
    """
    Plots a multi-objective line chart with true Viridis coloring based on sum of objectives.
    """

    num_objectives = len(objectives[0])
    num_configs = len(objectives)
    axis_names = axis_names or [f"Objective {i+1}" for i in range(num_objectives)]

    # Compute sum of objectives and normalize
    color_values = np.array([sum(obj) for obj in objectives])
    min_val, max_val = color_values.min(), color_values.max()
    norm_vals = (color_values - min_val) / (max_val - min_val + 1e-9)

    # Get Viridis colors
    viridis_colors = px.colors.sequential.Viridis
    color_map = [px.colors.sample_colorscale(viridis_colors, [v])[0] for v in norm_vals]

    fig = go.Figure()

    # Add each configuration as a line
    for i, obj_values in enumerate(objectives):
        fig.add_trace(
            go.Scatter(
                x=axis_names,
                y=obj_values,
                mode="lines+markers",
                name=str(theta[i]),
                showlegend=False,
                line=dict(color=color_map[i], width=2),
                marker=dict(size=8, color=color_map[i]),
                hovertemplate=(
                    f"<b>{theta[i]}</b><br>"
                    + "<br>".join(
                        [
                            f"{axis_names[j]}: {obj_values[j]:.2f}"
                            for j in range(num_objectives)
                        ]
                    )
                    + f"<br>Sum: {color_values[i]:.2f}<extra></extra>"
                ),
            )
        )

    # Build custom colorbar manually
    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        showlegend=False,
        marker=dict(
            colorscale="Viridis",
            cmin=min_val,
            cmax=max_val,
            color=[min_val, max_val],
            colorbar=dict(title="Sum of losses (darker = better)"),
            showscale=True,
        ),
        hoverinfo="none",
    )
    fig.add_trace(colorbar_trace)

    if baseline_results:
        baseline_colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
        color_idx = 0
        
        for model_name, result in baseline_results.items():
            if 'fairobs' in result:
                # fairobs is a list of [dp, ep, eo, tpr, fpr] for each sensitive attribute
                # Average across sensitive attributes or use first one
                fairobs = result['fairobs']
                if fairobs:  # Check if fairobs is not empty
                    # Average across all sensitive attributes
                    avg_fairobs = np.mean(fairobs, axis=0)  # [avg_dp, avg_ep, avg_eo, avg_tpr, avg_fpr]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=axis_names,
                            y=avg_fairobs,
                            mode="lines+markers",
                            name=model_name,
                            legend="legend2",
                            showlegend=True,  # Add this line explicitly
                            line=dict(color=baseline_colors[color_idx % len(baseline_colors)], width=3, dash='dash'),
                            marker=dict(size=10, color=baseline_colors[color_idx % len(baseline_colors)], symbol='square'),
                            hovertemplate=(
                                f"<b>{model_name}</b><br>"
                                + "<br>".join([
                                    f"{axis_names[j]}: {avg_fairobs[j]:.3f}"
                                    for j in range(len(avg_fairobs))
                                ])
                                + f"<br>Sum: {sum(avg_fairobs):.3f}<extra></extra>"
                            ),
                        )
                    )
                    color_idx += 1

    # Wrapped title
    title_wrap = 100
    wrapped_title = "<br>".join(textwrap.wrap(title, width=title_wrap))

    # Layout
    fig.update_layout(
        title=dict(
            text=f"{criteria}-objectives<br>{wrapped_title}",
            font=dict(size=12),
            y=0.95,  # Move title higher (default is ~0.9)
        ),
        xaxis_title="Objectives",
        yaxis_title="Values",
        plot_bgcolor="#F5E8D8",
        paper_bgcolor="#F5E8D8",  
        font=dict(size=15,color="#2B2B2B"),
        annotations=[
            go.layout.Annotation(
                showarrow=False,
                text=desc,
                x=0,
                y=1.35,  # Keep it above plot
                xref="paper",
                yref="paper",
                font=dict(size=15, color="#2B2B2B"),
                bgcolor= "#EBDDCB",  #"rgba(112,128,144,1)",
                bordercolor="black",
                borderwidth=2,
                borderpad=10,
                align="left",
            )
        ],
        margin=dict(t=160),  # Adds space at the top for annotation + title
    )

    if baseline_results:
        fig.update_layout(
            # showlegend=False,  # Hide the default legend since main traces don't use it
            legend2=dict(
                title=dict(text="Baseline Models"),
                x=0.5,
                xanchor="center",
                y=-0.35,
                yanchor="top",
                orientation="h",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1
            ),
            margin=dict(t=160, b=100)  # Add bottom margin for legend space
        )
    else:
        fig.update_layout(showlegend=False)

    if html:
        return fig.to_html(full_html=True, include_plotlyjs=True)
    else:
        fig.show()
