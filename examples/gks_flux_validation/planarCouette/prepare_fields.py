#!/usr/bin/env python3
"""Prepare impulsively started Couette flow: quiescent gas, moving top wall."""

from pathlib import Path


case = Path(__file__).resolve().parent
text = '''FoamFile
{
    version 2.0;
    format ascii;
    class volVectorField;
    location "0";
    object U;
}
dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
    xMin { type symmetryPlane; }
    xMax { type symmetryPlane; }
    bottom { type fixedValue; value uniform (0 0 0); }
    top { type fixedValue; value uniform (1 0 0); }
    zEmpty { type empty; }
}
'''
(case / "0" / "U").write_text(text, encoding="utf-8")
