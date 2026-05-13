import os


def configure_matplotlib_fonts():
    """Configure matplotlib to render Chinese labels when a CJK font is available."""
    try:
        import matplotlib
        from matplotlib import font_manager

        matplotlib.rcParams["axes.unicode_minus"] = False

        windir = os.environ.get("WINDIR", r"C:\Windows")
        font_files = [
            os.path.join(windir, "Fonts", "msyh.ttc"),
            os.path.join(windir, "Fonts", "msyh.ttf"),
            os.path.join(windir, "Fonts", "simhei.ttf"),
            os.path.join(windir, "Fonts", "simsun.ttc"),
            os.path.join(windir, "Fonts", "NotoSansCJK-Regular.ttc"),
        ]
        for font_file in font_files:
            if os.path.exists(font_file):
                try:
                    font_manager.fontManager.addfont(font_file)
                    font_name = font_manager.FontProperties(fname=font_file).get_name()
                    matplotlib.rcParams["font.family"] = "sans-serif"
                    matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
                    return font_name
                except Exception:
                    continue

        preferred_names = [
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "Noto Sans CJK SC",
            "WenQuanYi Micro Hei",
            "Arial Unicode MS",
        ]
        available_names = {font.name for font in font_manager.fontManager.ttflist}
        for font_name in preferred_names:
            if font_name in available_names:
                matplotlib.rcParams["font.family"] = "sans-serif"
                matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
                return font_name
    except Exception:
        pass
    return None
