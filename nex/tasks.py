from invoke import task
from pathlib import Path

basepath = "/Users/michaeldeistler/Documents/phd/jaxley_experiments/paper/"
open_cmd = "open"

fig_names = {
    "1": "fig1_illustration",
    "2": "fig2_demonstration",
    "3": "fig3_l5pc",
    "4": "fig4_rgc",
    "6": "fig6_mnist",
}


@task
def convertFigures(c, fig):
    _convert_svg2pdf(c, fig)
    _convert_pdf2png(c, fig)


@task
def _convert_svg2pdf(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_svg2pdf(c, str(f + 1))
        return
    pathlist = Path("{bp}/{fn}/fig/".format(bp=basepath, fn=fig_names[fig])).glob(
        "*.svg"
    )
    for path in pathlist:
        c.run(f"/Applications/Inkscape.app/Contents/MacOS/inkscape {str(path)} --export-pdf={str(path)[:-4]}.pdf")


@task
def _convert_pdf2png(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_pdf2png(c, str(f + 1))
        return
    pathlist = Path("{bp}/{fn}/fig/".format(bp=basepath, fn=fig_names[fig])).glob(
        "*.pdf"
    )
    for path in pathlist:
        c.run(
            f"/Applications/Inkscape.app/Contents/MacOS/inkscape {str(path)} --export-png={str(path)[:-4]}.png -b 'white' --export-dpi=700"
        )
