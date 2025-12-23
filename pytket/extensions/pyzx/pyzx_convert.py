# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fractions import Fraction

from pytket.architecture import Architecture
from pytket.circuit import Circuit, Op, OpType, Qubit, UnitID
from pyzx.circuit import Circuit as pyzxCircuit
from pyzx.circuit import gates as zxGates
from pyzx.graph.graph import Graph as PyzxGraph
from pyzx.routing.architecture import Architecture as PyzxArc

_tk_to_pyzx_gates = {
    OpType.Rz: zxGates.ZPhase,
    OpType.Rx: zxGates.XPhase,
    OpType.X: zxGates.NOT,
    OpType.Z: zxGates.Z,
    OpType.S: zxGates.S,  # gate.adjoint == False
    OpType.Sdg: zxGates.S,  # gate.adjoint == True
    OpType.T: zxGates.T,  # gate.adjoint == False
    OpType.Tdg: zxGates.T,  # gate.adjoint == True
    OpType.CX: zxGates.CNOT,
    OpType.CZ: zxGates.CZ,
    OpType.H: zxGates.HAD,
    OpType.SWAP: zxGates.SWAP,
}

_pyzx_to_tk_gates: dict = {item[1]: item[0] for item in _tk_to_pyzx_gates.items()}

_parameterised_gates = {OpType.Rz, OpType.Rx}


def tk_to_pyzx(tkcircuit: Circuit, denominator_limit: int = 1000000) -> pyzxCircuit:
    """
    Convert a tket :py:class:`~pytket._tket.circuit.Circuit` to a
    :py:class:`pyzx.circuit.Circuit`.

    :param tkcircuit: A circuit to be converted
    :param denominator_limit: The limit for denominator size when converting
        floats to fractions. Smaller limits allow for correct representation of simple
        fractions with non-exact floating-point representations, while larger limits
        allow for more precise angles.

    :return: The converted circuit
    """
    if not tkcircuit.is_simple:
        raise Exception("Cannot convert a non-simple tket Circuit to PyZX")
    c = pyzxCircuit(tkcircuit.n_qubits)
    if tkcircuit.name:
        c.name = tkcircuit.name
    for command in tkcircuit:
        op = command.op
        if op.type not in _tk_to_pyzx_gates:
            raise Exception(
                "Cannot convert tket gate: "
                + str(op)
                + ", as the gate type is unrecognised."
            )
        gate_class = _tk_to_pyzx_gates[op.type]
        adjoint = op.type == OpType.Sdg or op.type == OpType.Tdg  # noqa: PLR1714
        qbs = [q.index[0] for q in command.args]
        gate: zxGates.Gate  # assignment
        n_params = len(op.params)
        if n_params == 1:
            phase = op.params[0]
            if not isinstance(phase, float):
                raise Exception(
                    "Cannot convert tket gate: "
                    + str(op)
                    + ", as it contains symbolic parameters."
                )
            phase = Fraction(phase).limit_denominator(denominator_limit)
            gate = gate_class(*qbs, phase=phase)
        elif n_params > 1:
            raise Exception(
                "Cannot convert tket gate: "
                + str(op)
                + ", as it contains multiple parameters."
            )
        elif adjoint:
            gate = gate_class(*qbs, adjoint=True)
        else:
            gate = gate_class(*qbs)
        c.add_gate(gate)
    return c


def pyzx_to_tk(pyzx_circ: pyzxCircuit) -> Circuit:
    """
    Convert a :py:class:`pyzx.circuit.Circuit` to a tket :py:class:`~pytket._tket.circuit.Circuit` .
    Run :py:meth:`pyzx.circuit.Circuit.to_basic_gates` before conversion.

    :param pyzx_circ: A circuit to be converted

    :return: The converted circuit
    """
    c = Circuit(pyzx_circ.qubits, name=pyzx_circ.name)
    for g in pyzx_circ.gates:
        if type(g) not in _pyzx_to_tk_gates:
            raise Exception(
                "Cannot parse PyZX gate of type " + g.name + "into tket Circuit"
            )
        op_type = _pyzx_to_tk_gates[type(g)]
        if hasattr(g, "control"):  # noqa: SIM108
            qbs = [g.control, g.target]  # type: ignore
        else:
            qbs = [g.target]  # type: ignore

        if op_type == OpType.Sdg and not g.adjoint:  # type: ignore
            op_type = OpType.S
        elif op_type == OpType.Tdg and not g.adjoint:  # type: ignore
            op_type = OpType.T

        if (
            hasattr(g, "print_phase")
            and hasattr(g, "phase")
            and op_type in _parameterised_gates
        ):
            op = Op.create(op_type, g.phase)
        else:
            op = Op.create(op_type)

        c.add_gate(Op=op, args=qbs)
    return c


def tk_to_pyzx_arc(pytket_arc: Architecture, pyzx_arc_name: str = "") -> PyzxArc:
    """
    Convert a pytket :py:class:`~pytket.architecture.Architecture` to a pyzx
    :py:class:`pyzx.routing.architecture.Architecture` .
    The conversion will remove all the node names and will
    keep them only integer named in the order they are given
    in the node set of `pytket_arc`.

    :param pytket_arc: A Architecture to be converted
    :param pyzx_arc_name: Name of the architecture in pyzx

    :return: The converted pyzx Architecture
    """

    arcgraph = PyzxGraph()
    vertices = arcgraph.add_vertices(len(pytket_arc.nodes))
    arc_dict = dict()  # noqa: C408

    for i, x in enumerate(pytket_arc.nodes):
        arc_dict[x] = i  # noqa: PERF403

    edges = [
        (vertices[arc_dict[v1]], vertices[arc_dict[v2]])
        for v1, v2 in pytket_arc.coupling
    ]

    arcgraph.add_edges(edges)

    pyzx_arc = PyzxArc(pyzx_arc_name, coupling_graph=arcgraph)

    return pyzx_arc  # noqa: RET504


def pyzx_to_tk_arc(pyzx_arc: PyzxArc) -> Architecture:
    """
    Convert a pyzx :py:class:`pyzx.routing.architecture.Architecture`
    to a pytket :py:class:`~pytket.architecture.Architecture` .

    :param pytket_arc: A Architecture to be converted

    :return: The converted pyzx Architecture
    """

    return Architecture([(int(s[0]), int(s[1])) for s in pyzx_arc.graph.edges()])


def tk_to_pyzx_placed_circ(
    pytket_circ: Circuit,
    pytket_arc: Architecture,
    denominator_limit: int = 1000000,
    pyzx_arc_name: str = "",
) -> tuple[PyzxArc, pyzxCircuit, dict[UnitID, UnitID]]:
    """
    Convert a (placed) tket :py:class:`~pytket._tket.circuit.Circuit` with
    a given :py:class:`~pytket.architecture.Architecture` to a
    :py:class:`pyzx.circuit.Circuit` and the
    :py:class:`pyzx.routing.architecture.Architecture`
    and a map to give the information for converting the
    pyzx circuit back to pytket circuit using :py:func:`~.pyzx_to_tk_placed_circ`
    assigning each of the circuit qubits a one of the architecture nodes

    :param pytket_circ: A circuit to be converted
    :param pytket_arc: Corresponding Architecture
    :param denominator_limit: The limit for denominator size when converting
        floats to fractions. Smaller limits allow for correct representation of simple
        fractions with non-exact floating-point representations, while larger limits
        allow for more precise angles.
    :param pyzx_arc_name: Name of the architecture in pyzx

    :return: Tuple containing generated :py:class:`pyzx.circuit.Circuit` ,
        :py:class:`pyzx.routing.architecture.Architecture` and
        a map to give the information for converting
        the pyzx circuit back to pytket circuit using :py:func:`~.pyzx_to_tk_placed_circ`
        assigning each of the circuit qubits a one of the architecture nodes

    """

    simple_circ = pytket_circ.copy()

    q_map = simple_circ.flatten_registers()

    inv_q_map = {v: k for k, v in q_map.items()}

    arcgraph = PyzxGraph()
    vertices = arcgraph.add_vertices(len(pytket_arc.nodes))
    arc_dict = dict()  # noqa: C408
    qubit_dict = dict()  # noqa: C408

    for i in range(len(pytket_arc.nodes)):
        q = Qubit(i)
        qubit_dict[q] = i

    for i, x in enumerate(pytket_arc.nodes):  # noqa: B007
        arc_dict[x] = qubit_dict[q_map[x]]  # type: ignore

    edges = [
        (vertices[arc_dict[v1]], vertices[arc_dict[v2]])
        for v1, v2 in pytket_arc.coupling
    ]

    arcgraph.add_edges(edges)

    pyzx_arc = PyzxArc(pyzx_arc_name, coupling_graph=arcgraph)

    pyzx_circ = tk_to_pyzx(simple_circ, denominator_limit)

    return (pyzx_arc, pyzx_circ, inv_q_map)


def pyzx_to_tk_placed_circ(
    pyzx_circ: pyzxCircuit, q_map: dict[UnitID, UnitID]
) -> Circuit:
    """
    Convert a :py:class:`pyzx.circuit.Circuit` and a placment map
    to a placed tket :py:class:`~pytket._tket.circuit.Circuit` .
    Run :py:meth:`pyzx.circuit.Circuit.to_basic_gates` before conversion.

    :param pyzx_circ: A circuit to be converted
    :param q_map: placment map to assign each of the qubits in the
        pyzx circuit to one of the architecture nodes. It is recommended
        to use here the map generated by :py:func:`~.tk_to_pyzx_placed_circ`


    :return: The converted pytket circuit
    """

    pytket_circ = pyzx_to_tk(pyzx_circ)

    pytket_circ.rename_units(q_map)

    return pytket_circ
