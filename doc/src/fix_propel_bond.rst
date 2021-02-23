.. index:: fix propel/bond

fix propel/bond command
=======================

Syntax
""""""

.. parsed-literal::

    fix ID group-ID propel/bond magnitude keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* propel/bond = name of this fix command
* magnitude = magnitude of the propulsion force
* zero or more keyword/value pairs may be appended
* keyword = *btypes* or *reverse*

    .. parsed-literal::

        *btypes* values = one or more bond types
        *reverse* values = mode period
            mode = *no* or *periodically* or *stochastically*
            period = reversal timescale (time units)

Examples
""""""""

.. code-block:: LAMMPS

    # active filaments with constant terminal velocity
    fix 1 filaments propel/bond polar 1.0
    fix 2 all       viscous 1.0

    # coarse-grained active nematic filaments
    fix 1 filaments propel/bond nematic 1.0 1.0

Description
"""""""""""

Related commands
""""""""""""""""

:doc:`fix propel/self <fix_propel_self>`, :doc:`fix addforce <fix_addforce>`

----------------
