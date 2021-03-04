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
* keyword = *btypes* or *flip* or *reverses* or *check*

    .. parsed-literal::

        *btypes* values = one or more bond types
        *flip* values = one or more bond types
        *reverses* values = Treverse seed
            Treverse = propulsion reversal time (time units)
            seed = random number seed to use for stochastic reversing (positive integer)
        *check* values = nevery
            nevery = how often to check for new molecules during a run (time steps)

Examples
""""""""

.. code-block:: LAMMPS

    # active filaments with constant terminal velocity
    fix propulsion  all propel/bond 1.0
    fix viscosity   all viscous 1.0

    # coarse-grained active nematic filaments
    fix propulsion  all propel/bond 1.0 reverse 1.0 31415

    # mixture of passive and active polar filaments
    fix propulsion  all propel/bond 1.0 btypes 2

Description
"""""""""""

Adds a propulsion force with a constant magnitude to each bond whose constituent
atoms are part of the group. The force is distributed to each atom of the bond
equally.

By default, the propulsion force is directed along the bond direction from the
atom with smaller ID to the atom with the larger ID. Applied to a filament
constructed from a linear chain of bonded atoms with increasing atom ID, this
gives rise to a polar active force similar, though not identical, to that used
in :ref:`(Prathyusha) <Prathyusha>` and :ref:`(Peterson) <Peterson>`.

The *btypes* keyword can be used to filter which bond types the active force is
applied to. Bond types can use range notation (e.g., i*j) to specify a set of
bond types. By default, all bond types are assumed to be active.

The *flip* keyword can be used to specify a set of bond types for which the
active force direction is flipped from the default; that is, the force will
be directed from the atom with the larger ID to the atom with the smaller ID.
This can be used to create more complex active structures.

When the keyword *reverses* is used, the propulsion force will stochastically
reverse direction with an average period *Treverse*. This style of active
forcing exhibits nematic symmetry over time scales much longer than *Treverse*,
making it suitable for simulations of active nematics, as used in
:ref:`(Shi) <Shi>` and :ref:`(Joshi) <Joshi>`. Note that the force is reversed
on a per-molecule basis, and not per-bond. Thus, stochastic reversing cannot be
enabled if the atom style does not support molecule identifiers.

When the *check* keyword is, this fix will check for new molecules every
*nevery* time steps. This is useful in conjunction with other fixes that create
molecules during a run, such as :doc:`fix deposit <fix_deposit>`. If the keyword
isn't used, or if *nevery* is set to 0, then the fix will never check for new
molecules. This keyword is ignored if the *reverses* keyword isn't used.

.. note::

    Since the net force depends on the number of bonds in a molecule, rather
    than the number of atoms, you may want to consider adjusting the propulsion
    force, :math:`f_p`, to account for this. For example, a rigid chain of
    :math:`N` atoms will have :math:`N_b = N - 1` bonds, leading to a net
    propulsion force of :math:`F = (N - 1) f_p / N`, which is dependent on chain
    the length. To avoid this, the propulsion force can be scaled up by the
    ratio of number of atoms to number of bonds within the molecule.

----------------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files <restart>`.

This fix is not imposed  during minimization.

Restrictions
""""""""""""

The *reverses* and *check* keywords can only be used with molecular atom styles.

Related commands
""""""""""""""""

:doc:`fix propel/self <fix_propel_self>`, :doc:`fix addforce <fix_addforce>`

----------------

.. _Shi:

**(Shi)** X.-q. Shi and Y.-q. Ma, Nat. Commun., 4, 3013 (2013).

.. _Prathyusha:

**(Prathyusha)** K. R. Prathyusha, S. Henkes, R. Sknepnek, Phys. Rev. E, 97, 022606 (2018).

.. _Joshi:

**(Joshi)** A. Joshi, E. Putzig, A. Baskaran and M. F. Hagan, Soft Matter, 15, 94 (2019).

.. _Peterson:

**(Peterson)** M. S. E. Peterson, M. F. Hagan, A. Baskaran, J. Stat. Mech, 013216 (2020).
