from manim import *
import numpy as np

class LagrangeInterpolationDualPane(Scene):
    def construct(self):
        # --- Configuración del Layout (Gráfica Izquierda / Texto Derecha) ---

        # 1. Gráfica y Ejes (Pane Izquierdo)
        # **CAMBIO CLAVE: Reducir las longitudes para dejar más espacio al texto.**
        axes = Axes(
            x_range=[-0.5, 5.5, 1],
            y_range=[-1, 5, 1],
            x_length=7,   # Reducido de 8
            y_length=4.5, # Reducido de 5
            axis_config={"color": GREY_B, "include_numbers": True},
        ).to_edge(LEFT, buff=0.7).shift(DOWN*0.2) # Mover a la izquierda y un poco hacia abajo para centrar visualmente
        
        labels = axes.get_axis_labels(x_label="x", y_label="y")

        # 2. Posición del Texto (Pane Derecho)
        text_center = RIGHT * 4 # Mantiene el centro de referencia a la derecha
        
        # Puntos de interpolación
        points_data = [(1, 2), (2, 3), (4, 1)]
        point_colors = [RED, BLUE, GREEN]

        dots_list = [Dot(axes.c2p(x, y), color=color) 
                for (x, y), color in zip(points_data, point_colors)]
        
        # Ajustamos el next_to de las etiquetas de los puntos ya que los dots se movieron
        dot_labels_list = [
            MathTex(f"P_{i}({x},{y})", color=color)
            .scale(0.7)
            .next_to(dot, UP if y > 2 else DOWN, buff=0.1)
            for i, ((x, y), dot, color) in enumerate(zip(points_data, dots_list, point_colors))
        ]
        
        dots = VGroup(*dots_list)
        dot_labels = VGroup(*dot_labels_list)

        # Presentación inicial
        self.play(Create(axes), Write(labels))
        self.play(
            LaggedStart(*[FadeIn(dot, scale=1.5) for dot in dots_list], lag_ratio=0.3),
            LaggedStart(*[Write(lbl) for lbl in dot_labels_list], lag_ratio=0.3),
            run_time=2
        )
        self.wait(1)
        
        # --- Sección 1: Polinomio Base de Lagrange (L_i) ---
        
        # FÓRMULA GENERAL DEL POLINOMIO BASE
        title_L = Text("1. Polinomio Base $L_i(x)$", font_size=36).move_to(text_center + UP * 3)
        L_general_formula = MathTex(
            "L_i(x) = \\prod_{j=0, j \\ne i}^{n} \\frac{x-x_j}{x_i-x_j}", 
            color=GOLD
        ).next_to(title_L, DOWN, buff=0.5)
        
        self.play(Write(title_L), Write(L_general_formula))
        self.wait(2)
        
        # Las funciones base específicas
        def L0(x): return ((x-2)*(x-4))/((1-2)*(1-4))
        def L1(x): return ((x-1)*(x-4))/((2-1)*(2-4))
        def L2(x): return ((x-1)*(x-2))/((4-1)*(4-2))
        
        L_functions = [L0, L1, L2]
        L_tex_strings = [
            "L_0(x) = \\frac{(x-2)(x-4)}{(1-2)(1-4)}",
            "L_1(x) = \\frac{(x-1)(x-4)}{(2-1)(2-4)}",
            "L_2(x) = \\frac{(x-1)(x-2)}{(4-1)(4-2)}"
        ]
        
        L_colors = [RED, BLUE, GREEN]
        L_graphs = [axes.plot(f, color=color) for f, color in zip(L_functions, L_colors)]
        
        all_L_graphs = VGroup()
        for i in range(len(points_data)):
            # La fórmula específica ahora tiene más espacio
            L_tex_specific = MathTex(L_tex_strings[i], color=L_colors[i]).next_to(L_general_formula, DOWN, buff=0.4)
            
            P_dot = dots_list[i]
            L_at_nodes = VGroup()
            for j in range(len(points_data)):
                y_val = L_functions[i](points_data[j][0])
                if i == j:
                    node_color = L_colors[i]
                    radius = 0.1
                else:
                    node_color = GREY
                    radius = 0.05
                L_at_nodes += Dot(axes.c2p(points_data[j][0], y_val), color=node_color, radius=radius)

            self.play(
                Write(L_tex_specific), 
                Create(L_graphs[i]),
                FadeIn(L_at_nodes),
                Indicate(P_dot, scale_factor=1.5, color=L_colors[i]),
                run_time=2
            )
            
            self.wait(1)
            all_L_graphs += L_graphs[i]
            self.play(FadeOut(L_at_nodes), FadeOut(L_tex_specific)) 
        
        self.play(
            *[g.animate.set_opacity(0.1) for g in all_L_graphs],
            FadeOut(title_L, L_general_formula) 
        )
        self.wait(1)
        
        # --- Sección 2: Polinomio Interpolador de Lagrange P(x) ---
        
        # FÓRMULA GENERAL DEL POLINOMIO INTERPOLADOR
        title_P = Text("2. Polinomio Interpolador $P(x)$", font_size=36).move_to(text_center + UP * 3)
        P_general_formula = MathTex(
            "P(x) = \\sum_{i=0}^{n} y_i L_i(x)",
            color=YELLOW
        ).next_to(title_P, DOWN, buff=0.5)
        
        self.play(Write(title_P), Write(P_general_formula))
        self.wait(2)

        # El polinomio completo
        def P(x): return points_data[0][1]*L0(x) + points_data[1][1]*L1(x) + points_data[2][1]*L2(x)
        P_graph = axes.plot(P, color=YELLOW, stroke_width=4)
        
        # La expresión específica (debajo de la general)
        P_tex_val = MathTex(
            "P(x) = 2 L_0(x) + 3 L_1(x) + 1 L_2(x)",
            color=YELLOW
        ).next_to(P_general_formula, DOWN, buff=0.4) # Ajustamos el buff aquí
        
        # Animación de las contribuciones
        contribution_terms = VGroup()
        for i, (x, y) in enumerate(points_data):
            def term_func(x): 
                return y * L_functions[i](x)
            
            term_graph = axes.plot(term_func, color=L_colors[i], stroke_width=2.5)
            
            # Reposicionamos las etiquetas y_i para que no interfieran con P_tex_val
            y_tex = MathTex(f"y_{i} = {y}", color=L_colors[i]).scale(0.8)
            y_tex.move_to(text_center + DOWN*0.5 + RIGHT*(i-1)*1.5)

            self.play(
                Indicate(dots_list[i], scale_factor=1.5, color=L_colors[i]),
                Write(y_tex),
                Create(term_graph),
                run_time=1.5
            )
            contribution_terms += term_graph
            
            self.play(FadeOut(y_tex))

        # Mostramos la fórmula específica de la suma
        self.play(Write(P_tex_val))
        self.wait(1)
        
        final_dots_list = dots.submobjects
        self.play(
            Transform(contribution_terms, P_graph),
            LaggedStart(*[Flash(dot, color=YELLOW, flash_radius=0.3) for dot in final_dots_list], lag_ratio=0.1),
            run_time=3
        )
        
        final_text = Text("El Polinomio P(x) interpola todos los puntos.", font_size=28).next_to(P_tex_val, DOWN, buff=0.6) # Ajustamos el buff aquí
        self.play(Write(final_text))
        self.wait(3)
        
        # Limpieza final
        self.play(
            FadeOut(
                axes, labels, dots, dot_labels, 
                title_P, P_general_formula, P_tex_val, final_text, 
                contribution_terms, all_L_graphs
            )
        )
        self.wait(1)
