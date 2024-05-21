from manim import *


class HelloLaTeX(Scene):

  def construct(self):


    text = Text("Activation Functions").to_corner(UL)


    self.add(text)

    tex = MathTex(r"20x \cdot \beta", font_size=100)
    self.add(tex)

    axes = Axes(

                y_range = [0, 5],
                x_range = [-5, 5],
                x_length = 10
    )

    axes.add_coordinates()

    self.play(Write(axes))
    self.wait(1)

    plot = axes.plot(lambda x: x if x>0 else 0, color=RED)

    self.play(Write(plot))
    self.wait(5)