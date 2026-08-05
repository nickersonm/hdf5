package core

import "testing"

// TestMessageTypesMatchTheSpecification pins every object header message type against the numbers
// in the HDF5 File Format Specification, section IV.A.2.
//
// These constants are used by the reader and the writer alike, so one that is wrong is invisible
// to any round-trip test: this package writes a message under the wrong number, reads it back
// under the same wrong number, and agrees with itself perfectly while producing a file no other
// implementation can interpret. MsgAttributeInfo shipped as 15 for exactly that reason -- 15 is
// Shared Message Table, Attribute Info is 21 -- and the effect was that every attribute written
// to dense storage was invisible to the reference library, with no error anywhere.
//
// A table like this is worth more than it looks. It is the only check in the package that can
// fail on a value the rest of the package is consistently wrong about.
func TestMessageTypesMatchTheSpecification(t *testing.T) {
	for _, c := range []struct {
		got  MessageType
		want MessageType
		name string
	}{
		{MsgNil, 0x0000, "NIL"},
		{MsgDataspace, 0x0001, "Dataspace"},
		{MsgLinkInfo, 0x0002, "Link Info"},
		{MsgDatatype, 0x0003, "Datatype"},
		{MsgFillValueOld, 0x0004, "Fill Value (old)"},
		{MsgFillValue, 0x0005, "Fill Value"},
		{MsgLinkMessage, 0x0006, "Link"},
		{MsgDataLayout, 0x0008, "Data Layout"},
		{MsgFilterPipeline, 0x000B, "Filter Pipeline"},
		{MsgAttribute, 0x000C, "Attribute"},
		{MsgName, 0x000D, "Object Comment"},
		{MsgContinuation, 0x0010, "Object Header Continuation"},
		{MsgSymbolTable, 0x0011, "Symbol Table"},
		{MsgAttributeInfo, 0x0015, "Attribute Info"},
		{MsgRefCount, 0x0016, "Object Reference Count"},
	} {
		if c.got != c.want {
			t.Errorf("%s message type is %d (0x%04X), the specification says %d (0x%04X)", c.name, c.got, uint16(c.got), c.want, uint16(c.want))
		}
	}
}
