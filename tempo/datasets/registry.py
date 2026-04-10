from __future__ import annotations

from .sources import cameo, cremad, emodb, emotale, iemocap, ravdess, savee, tess

SOURCES = {
    "emodb": emodb,
    "iemocap": iemocap,
    "cremad": cremad,
    "savee": savee,
    "tess": tess,
    "ravdess": ravdess,
    "emotale": emotale,
    "cameo": cameo,
}

