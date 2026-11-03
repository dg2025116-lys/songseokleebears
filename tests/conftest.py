import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _korean_font():
    """그림 테스트에서 한글 글리프 경고가 쏟아지지 않도록 폰트를 한 번 설정한다."""
    import matplotlib
    matplotlib.use("Agg")
    from kbo_heatmap import plots
    plots.setup_font()
