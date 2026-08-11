package hdf5_test

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/scigolib/hdf5"
)

// A user block is the space the format specification reserves before the superblock so that an
// HDF5 file can be wrapped in another format, or carry a descriptive header, without the library
// losing track of the objects inside it. The superblock is located by searching byte offset 0,
// then 512, then each successive power-of-two multiple.
//
// v0.14.0 read byte offset 0 and nothing else, so every file with a user block was rejected as
// "not an HDF5 file". The case that makes this matter in practice is MATLAB: a .mat file saved
// with -v7.3 is an HDF5 file behind a 512-byte ASCII header, so none of them could be opened.
//
// These tests build the files with h5py rather than with this package, deliberately. A file this
// package wrote has no user block, so a round trip through its own writer cannot reach the defect
// at all -- the same blind spot that let five interop defects ship in v0.14.0.

// writeWithH5py builds a file with the given user block size and two float64 datasets, one of them
// shaped like a column vector saved from a column-major producer.
func writeWithH5py(t *testing.T, path string, userBlockSize int) {
	t.Helper()
	const script = `
import h5py, sys, numpy as np
path, ub = sys.argv[1], int(sys.argv[2])
kw = {"userblock_size": ub} if ub else {}
with h5py.File(path, "w", **kw) as f:
    f.create_dataset("row", data=np.array([[1.5, 2.5, 3.5, 4.5]], dtype="<f8"))
    f.create_dataset("column", data=np.array([[1.0], [2.0], [3.0]], dtype="<f8"))
if ub:
    with open(path, "r+b") as fh:
        fh.write(b"MATLAB 7.3 MAT-file, written for a test")
`
	cmd := exec.Command("python3", "-c", script, path, itoa(userBlockSize))
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("h5py could not write %s: %v\n%s", filepath.Base(path), err, strings.TrimSpace(string(out)))
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// TestAFileWithAUserBlockOpens is the defect this file exists for: the superblock is not at byte
// zero and must still be found.
func TestAFileWithAUserBlockOpens(t *testing.T) {
	requireH5py(t)
	for _, userBlock := range []int{0, 512, 1024, 2048} {
		t.Run("userblock_"+itoa(userBlock), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "file.h5")
			writeWithH5py(t, path, userBlock)

			// The signature really is where the test claims, or the case being covered is not the one described.
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if got := strings.Index(string(raw), "\x89HDF\r\n\x1a\n"); got != userBlock {
				t.Fatalf("the signature is at offset %d, want %d; the fixture is not exercising a user block", got, userBlock)
			}

			f, err := hdf5.Open(path)
			if err != nil {
				t.Fatalf("a file with a %d-byte user block did not open: %v", userBlock, err)
			}
			defer f.Close()

			var seen int
			f.Walk(func(path string, obj hdf5.Object) {
				d, ok := obj.(*hdf5.Dataset)
				if !ok {
					return
				}
				seen++
				// Values must survive the offset, not merely the open: an address applied against the wrong base reads plausible garbage rather than failing.
				vals, err := d.Read()
				if err != nil {
					t.Fatalf("%s: Read: %v", path, err)
				}
				switch d.Name() {
				case "row":
					want := []float64{1.5, 2.5, 3.5, 4.5}
					if len(vals) != len(want) {
						t.Fatalf("row has %d values, want %d", len(vals), len(want))
					}
					for i := range want {
						if math.Abs(vals[i]-want[i]) > 1e-12 {
							t.Errorf("row[%d] = %g, want %g", i, vals[i], want[i])
						}
					}
				case "column":
					if len(vals) != 3 {
						t.Errorf("column has %d values, want 3", len(vals))
					}
				}
			})
			if seen != 2 {
				t.Errorf("walked %d datasets, want 2", seen)
			}
		})
	}
}

// TestNonHDF5FilesAreStillRejected pins that the search does not turn the signature check into a
// scan that accepts anything with those eight bytes somewhere in it. Only the permitted offsets
// count.
func TestNonHDF5FilesAreStillRejected(t *testing.T) {
	dir := t.TempDir()
	cases := map[string][]byte{
		"empty":                   {},
		"short":                   []byte("no"),
		"plain text":              []byte(strings.Repeat("not an hdf5 file at all. ", 100)),
		"signature at offset 100": append(append(make([]byte, 100), []byte("\x89HDF\r\n\x1a\n")...), make([]byte, 1000)...),
		"signature at offset 700": append(append(make([]byte, 700), []byte("\x89HDF\r\n\x1a\n")...), make([]byte, 1000)...),
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(dir, strings.ReplaceAll(name, " ", "_"))
			if err := os.WriteFile(path, body, 0o600); err != nil {
				t.Fatal(err)
			}
			f, err := hdf5.Open(path)
			if err == nil {
				f.Close()
				t.Errorf("%s was accepted as an HDF5 file", name)
			}
		})
	}
}

// TestDimsReportsTheDataspaceShape covers the second thing a caller outside this module could not
// previously learn: Read returns a flat slice, so without the shape there is no way to tell a
// 1-by-N array from an N-by-1 one. That distinction is exactly what a column-major producer
// depends on.
func TestDimsReportsTheDataspaceShape(t *testing.T) {
	requireH5py(t)
	path := filepath.Join(t.TempDir(), "shapes.h5")
	writeWithH5py(t, path, 0)

	f, err := hdf5.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	want := map[string][]uint64{
		"row":    {1, 4},
		"column": {3, 1},
	}
	seen := 0
	f.Walk(func(_ string, obj hdf5.Object) {
		d, ok := obj.(*hdf5.Dataset)
		if !ok {
			return
		}
		expect, known := want[d.Name()]
		if !known {
			return
		}
		seen++
		got, err := d.Dims()
		if err != nil {
			t.Fatalf("%s: Dims: %v", d.Name(), err)
		}
		if len(got) != len(expect) {
			t.Fatalf("%s has %d dimensions %v, want %d %v", d.Name(), len(got), got, len(expect), expect)
		}
		for i := range expect {
			if got[i] != expect[i] {
				t.Errorf("%s dims = %v, want %v", d.Name(), got, expect)
				break
			}
		}
	})
	if seen != len(want) {
		t.Errorf("checked %d datasets, want %d", seen, len(want))
	}
}
