package hdf5_test

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/scigolib/hdf5"
)

// Dense attribute storage was wrong in both directions, and the two mistakes cancelled: the writer
// stamped its B-tree as a link name index and the reader read every B-tree as one, so a round trip
// through this package agreed with itself and matched nothing the reference library writes.
//
// Both tests here therefore cross the boundary. A Go round trip cannot replace either of them, and
// that is the entire point: it is what let the defect sit behind a passing suite.
//
// The counts straddle the compact-to-dense transition deliberately. Four stays compact and is the
// regression risk; six and above go dense, and six is where libhdf5 used to fail inside its metadata
// cache rather than at the record.

// TestDenseAttributesAreReadableByTheReferenceLibrary writes attributes here and reads them there.
func TestDenseAttributesAreReadableByTheReferenceLibrary(t *testing.T) {
	requireH5py(t)
	for _, n := range []int{4, 6, 9, 20, 50} {
		t.Run(fmt.Sprintf("%d_attributes", n), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "dense.h5")
			fw, err := hdf5.CreateForWrite(path, hdf5.CreateTruncate)
			if err != nil {
				t.Fatalf("CreateForWrite: %v", err)
			}
			g, err := fw.CreateGroup("/metadata")
			if err != nil {
				t.Fatalf("CreateGroup: %v", err)
			}
			for i := 0; i < n; i++ {
				if err := g.WriteAttribute(fmt.Sprintf("a%03d", i), float64(i)); err != nil {
					t.Fatalf("WriteAttribute %d: %v", i, err)
				}
			}
			if err := fw.Close(); err != nil {
				t.Fatalf("Close: %v", err)
			}

			got := readWithH5py(t, path)
			attrs, ok := got["attrs"].(map[string]interface{})
			if !ok {
				t.Fatalf("h5py returned no attribute map: %v", got)
			}
			meta, ok := attrs["metadata"].(map[string]interface{})
			if !ok {
				t.Fatalf("h5py saw no /metadata attributes: %v", attrs)
			}
			if len(meta) != n {
				t.Fatalf("the reference library sees %d attributes, want %d", len(meta), n)
			}
			// Values, not just names: a heap ID resolved against the wrong base reads plausible numbers.
			for i := 0; i < n; i++ {
				name := fmt.Sprintf("a%03d", i)
				v, present := meta[name]
				if !present {
					t.Fatalf("%s is missing", name)
				}
				// WriteAttribute emits a one-element array rather than a scalar dataspace, so the
				// reference library reports [i] and not i. That is this package's existing shape choice
				// and not what these tests are about; accept either.
				if got, ok := scalarOrSingleton(v); !ok || math.Abs(got-float64(i)) > 1e-12 {
					t.Errorf("%s = %v, want %d", name, v, i)
				}
			}
		})
	}
}

// TestDenseAttributesWrittenByTheReferenceLibraryAreReadable is the other direction, and it needs
// libver='latest': with the default library version bounds the reference library keeps attributes
// compact however many there are, so a fixture written without it exercises no dense storage at all
// and passes for the wrong reason.
func TestDenseAttributesWrittenByTheReferenceLibraryAreReadable(t *testing.T) {
	requireH5py(t)
	const n = 20
	path := filepath.Join(t.TempDir(), "libhdf5-dense.h5")

	script := fmt.Sprintf(`
import h5py, numpy as np, sys
with h5py.File(sys.argv[1], "w", libver="latest") as f:
    d = f.create_dataset("x", data=np.arange(4.0))
    for i in range(%d):
        d.attrs.create("a%%03d" %% i, np.float64(i))
`, n)
	if out, err := exec.Command("python3", "-c", script, path).CombinedOutput(); err != nil {
		t.Fatalf("h5py could not write the fixture: %v\n%s", err, strings.TrimSpace(string(out)))
	}

	// The fixture really is dense, or this test covers the compact path by accident.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "FRHP") {
		t.Fatal("the fixture has no fractal heap, so its attributes are compact and this test proves nothing")
	}

	f, err := hdf5.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	checked := 0
	f.Walk(func(p string, obj hdf5.Object) {
		d, ok := obj.(*hdf5.Dataset)
		if !ok {
			return
		}
		names, err := d.ListAttributes()
		if err != nil {
			t.Fatalf("%s: ListAttributes: %v", p, err)
		}
		if len(names) != n {
			t.Fatalf("%s: read %d attributes, want %d", p, len(names), n)
		}
		for i := 0; i < n; i++ {
			name := fmt.Sprintf("a%03d", i)
			v, err := d.ReadAttribute(name)
			if err != nil {
				t.Fatalf("%s: ReadAttribute(%s): %v", p, name, err)
			}
			got, ok := v.(float64)
			if !ok || math.Abs(got-float64(i)) > 1e-12 {
				t.Errorf("%s = %v, want %d", name, v, i)
			}
		}
		checked++
	})
	if checked != 1 {
		t.Errorf("checked %d datasets, want 1", checked)
	}
}

// TestAnObjectWithNoAttributesIsNotAnError guards the other half of the reader change. Attribute
// parse failures are now returned rather than discarded, and the easy way to get that wrong is to
// turn "nothing to read" into a failure.
func TestAnObjectWithNoAttributesIsNotAnError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bare.h5")
	fw, err := hdf5.CreateForWrite(path, hdf5.CreateTruncate)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fw.CreateGroup("/bare"); err != nil {
		t.Fatal(err)
	}
	if err := fw.Close(); err != nil {
		t.Fatal(err)
	}

	f, err := hdf5.Open(path)
	if err != nil {
		t.Fatalf("a file whose groups have no attributes must open: %v", err)
	}
	defer f.Close()
	for _, ch := range f.Root().Children() {
		g, ok := ch.(*hdf5.Group)
		if !ok {
			continue
		}
		attrs, err := g.Attributes()
		if err != nil {
			t.Errorf("%s: Attributes: %v", g.Name(), err)
		}
		if len(attrs) != 0 {
			t.Errorf("%s: got %d attributes, want none", g.Name(), len(attrs))
		}
	}
}

// scalarOrSingleton reads a value the reference library reported as either a scalar or a
// one-element array.
func scalarOrSingleton(v interface{}) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case []interface{}:
		if len(t) != 1 {
			return 0, false
		}
		f, ok := t[0].(float64)
		return f, ok
	}
	return 0, false
}
