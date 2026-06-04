@ECHO OFF
pushd %~dp0

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)

%SPHINXBUILD% -b html . _build\html %SPHINXOPTS%
popd
