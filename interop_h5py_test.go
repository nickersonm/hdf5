package hdf5_test

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/scigolib/hdf5"
)

// The reference HDF5 library is the only authority on whether a file this package writes is a
// valid HDF5 file. Every other test here is a round trip through this package's own reader, and a
// round trip cannot see a constant that the reader and the writer are wrong about together.
//
// That is not hypothetical. Three defects shipped in v0.14.0 behind exactly that blind spot:
// the Attribute Info message type was 15 where the specification says 21, float attributes wrote
// a class bit field of zero which puts the sign bit on top of the mantissa, and the filter
// pipeline message was stamped version 2 while carrying the version 1 layout. All three round
// tripped perfectly and none produced a file the reference library could read.
//
// testdata/c-library-corpus checks the other direction, that this package can read what the C
// library wrote. This is the missing half.
//
// Skips when h5py is unavailable so it does not break a machine without it; set
// HDF5_REQUIRE_H5PY=1 to turn the skip into a failure, which is what CI and a release check want.
func requireH5py(t *testing.T) {
	t.Helper()
	if err := exec.Command("python3", "-c", "import h5py").Run(); err != nil {
		if os.Getenv("HDF5_REQUIRE_H5PY") != "" {
			t.Fatalf("h5py is unavailable and HDF5_REQUIRE_H5PY is set: %v", err)
		}
		t.Skip("h5py not available; set HDF5_REQUIRE_H5PY=1 to make this a failure")
	}
}

// readWithH5py returns what the reference library sees in the file: every dataset by path, and
// every attribute of every group and dataset.
func readWithH5py(t *testing.T, path string) map[string]interface{} {
	t.Helper()
	const script = `
import h5py, json, sys, numpy as np
out = {"datasets": {}, "attrs": {}}
def norm(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, bytes): return v.decode()
    return v
with h5py.File(sys.argv[1], "r") as f:
    def visit(name, obj):
        out["attrs"][name] = {k: norm(v) for k, v in obj.attrs.items()}
        if isinstance(obj, h5py.Dataset):
            out["datasets"][name] = norm(obj[...])
    f.visititems(visit)
    out["attrs"]["/"] = {k: norm(v) for k, v in f.attrs.items()}
print(json.dumps(out))
`
	cmd := exec.Command("python3", "-c", script, path)
	stdout, err := cmd.Output()
	if err != nil {
		var stderr string
		if ee, ok := err.(*exec.ExitError); ok {
			stderr = string(ee.Stderr)
		}
		t.Fatalf("the reference library could not read %s: %v\n%s", filepath.Base(path), err, strings.TrimSpace(stderr))
	}
	var got map[string]interface{}
	if err := json.Unmarshal(stdout, &got); err != nil {
		t.Fatalf("decoding h5py output: %v", err)
	}
	return got
}

// TestAttributesAreReadableByTheReferenceLibrary covers the compact attribute path, which is what
// an object with a handful of attributes uses.
func TestAttributesAreReadableByTheReferenceLibrary(t *testing.T) {
	requireH5py(t)
	path := filepath.Join(t.TempDir(), "attrs.h5")

	fw, err := hdf5.CreateForWrite(path, hdf5.CreateTruncate)
	if err != nil {
		t.Fatal(err)
	}
	g, err := fw.CreateGroup("/metadata")
	if err != nil {
		t.Fatal(err)
	}
	// A float and an int, because they take different paths and the float one was broken.
	if err := g.WriteAttribute("p_tx_dbm", 30.5); err != nil {
		t.Fatal(err)
	}
	if err := g.WriteAttribute("num_apertures", int64(2)); err != nil {
		t.Fatal(err)
	}
	ds, err := fw.CreateDataset("/x", hdf5.Float64, []uint64{3})
	if err != nil {
		t.Fatal(err)
	}
	if err := ds.Write([]float64{1.5, 2.5, 3.5}); err != nil {
		t.Fatal(err)
	}
	ds.Close()
	if err := fw.Close(); err != nil {
		t.Fatal(err)
	}

	got := readWithH5py(t, path)
	attrs, _ := got["attrs"].(map[string]interface{})
	meta, ok := attrs["metadata"].(map[string]interface{})
	if !ok {
		t.Fatalf("the reference library reports no /metadata group; it saw %v", keysOf(attrs))
	}
	if len(meta) != 2 {
		t.Fatalf("the reference library reads %d attributes on /metadata, want 2: %v", len(meta), meta)
	}
	if v := firstFloat(meta["p_tx_dbm"]); v != 30.5 {
		t.Errorf("p_tx_dbm reads %v, want 30.5", meta["p_tx_dbm"])
	}
	if v := firstFloat(meta["num_apertures"]); v != 2 {
		t.Errorf("num_apertures reads %v, want 2", meta["num_apertures"])
	}
}

// TestDatasetsAreReadableByTheReferenceLibrary is the baseline: if this fails, nothing above it
// means anything.
func TestDatasetsAreReadableByTheReferenceLibrary(t *testing.T) {
	requireH5py(t)
	path := filepath.Join(t.TempDir(), "data.h5")

	fw, err := hdf5.CreateForWrite(path, hdf5.CreateTruncate)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{-1.25, 0, 1e-9, 1e9}
	ds, err := fw.CreateDataset("/values", hdf5.Float64, []uint64{uint64(len(want))})
	if err != nil {
		t.Fatal(err)
	}
	if err := ds.Write(want); err != nil {
		t.Fatal(err)
	}
	ds.Close()
	if err := fw.Close(); err != nil {
		t.Fatal(err)
	}

	got := readWithH5py(t, path)
	sets, _ := got["datasets"].(map[string]interface{})
	vals, ok := sets["values"].([]interface{})
	if !ok {
		t.Fatalf("the reference library reports no /values dataset; it saw %v", keysOf(sets))
	}
	if len(vals) != len(want) {
		t.Fatalf("read %d values, want %d", len(vals), len(want))
	}
	for i := range want {
		if v, _ := vals[i].(float64); v != want[i] {
			t.Errorf("value %d reads %v, want %v", i, vals[i], want[i])
		}
	}
}

func keysOf(m map[string]interface{}) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// firstFloat reads a scalar that h5py may return as a bare number or as a one-element array.
func firstFloat(v interface{}) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case []interface{}:
		if len(t) > 0 {
			if f, ok := t[0].(float64); ok {
				return f
			}
		}
	}
	return 0
}

// TestCompressedDatasetsAreReadableByTheReferenceLibrary covers the filter pipeline.
//
// WithGZIPCompression worked spectacularly on the write side and produced files nothing could
// open: the pipeline message was stamped version 2 while carrying the version 1 layout, so a
// conforming reader started the first filter at byte 2, read the six version 1 reserved bytes as
// the filter identifier, and got filter ID 0. Version 1 also pads the client-data section to a
// multiple of eight, which deflate's single level value makes the common case.
//
// Two further defects were behind that one, and both were found by this test rather than by any
// round trip: version 1 stores the filter name length ALREADY PADDED to a multiple of eight, and
// HDF5 filter 1 is DEFLATE in the ZLIB container rather than in the GZIP one that compress/gzip
// writes. Each produced a different, precise complaint from the reference library.
//
// Compressing well is not the property worth testing. Being readable is -- so this asserts both,
// and the size check doubles as the guard against the filter being silently skipped.
func TestCompressedDatasetsAreReadableByTheReferenceLibrary(t *testing.T) {
	requireH5py(t)
	path := filepath.Join(t.TempDir(), "gzip.h5")

	// Smooth and repetitive, so the filter has something to find and a failure to compress is
	// visible as well as a failure to decompress.
	want := make([]float64, 4096)
	for i := range want {
		want[i] = float64(i%64) * 0.5
	}

	fw, err := hdf5.CreateForWrite(path, hdf5.CreateTruncate)
	if err != nil {
		t.Fatal(err)
	}
	// Chunking is not optional: HDF5 filters only apply to chunked storage, and WithGZIPCompression
	// on a contiguous dataset is silently ignored -- the file is written uncompressed with no
	// filter pipeline and no error. The size assertion below is what catches that.
	ds, err := fw.CreateDataset("/compressed", hdf5.Float64, []uint64{uint64(len(want))},
		hdf5.WithChunkDims([]uint64{512}), hdf5.WithGZIPCompression(6))
	if err != nil {
		t.Fatal(err)
	}
	if err := ds.Write(want); err != nil {
		t.Fatal(err)
	}
	ds.Close()
	if err := fw.Close(); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	raw := int64(len(want) * 8)
	if info.Size() >= raw {
		t.Errorf("the file is %d bytes against %d bytes of raw data; nothing was compressed", info.Size(), raw)
	}

	got := readWithH5py(t, path)
	sets, _ := got["datasets"].(map[string]interface{})
	vals, ok := sets["compressed"].([]interface{})
	if !ok {
		t.Fatalf("the reference library reports no /compressed dataset; it saw %v", keysOf(sets))
	}
	if len(vals) != len(want) {
		t.Fatalf("read %d values, want %d", len(vals), len(want))
	}
	for i := range want {
		if v, _ := vals[i].(float64); v != want[i] {
			t.Fatalf("value %d reads %v, want %v", i, vals[i], want[i])
		}
	}
	t.Logf("%d bytes on disk against %d raw, and every value round-trips through the reference library", info.Size(), raw)
}
