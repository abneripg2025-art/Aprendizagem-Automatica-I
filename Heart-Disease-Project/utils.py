def num_groups(quantity, quantity_per_group):
    num_groups = quantity // quantity_per_group
    if quantity % quantity_per_group > 0:
        num_groups += 1

    return num_groups

def show_variable_plots(variables, plot_function, max_rows, max_cols):
    num_variables = len(variables)
    rows = num_groups(num_variables, max_cols)
    columns = min(num_variables, max_cols)
    rows = min(rows, max_rows)

    max_graphics_plot = rows * columns

    if num_variables > max_graphics_plot:
        number_pages = num_groups(num_variables, max_graphics_plot)

        for i in range(0, number_pages):
            variables_to_show = variables[0:max_graphics_plot]
            variables = variables[max_graphics_plot:]
            plot_function(variables_to_show, rows, columns)
    else:
        plot_function(variables, rows, columns)

