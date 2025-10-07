# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['face_registration_standalone.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\cv2\\data', 'cv2\\data')],
    hiddenimports=['psycopg2', 'pickle'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Face_Registration_Tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
