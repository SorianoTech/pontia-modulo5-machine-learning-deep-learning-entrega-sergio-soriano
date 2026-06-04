Uso de la documentación
=======================

La documentación HTML se genera desde la raíz del proyecto:

.. code-block:: powershell

   .venv\Scripts\activate
   sphinx-build -b html docs docs\_build\html

El resultado queda disponible en ``docs\_build\html\index.html``.
