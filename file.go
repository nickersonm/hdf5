// Package hdf5 provides a pure Go implementation for reading HDF5 files.
// It supports HDF5 format versions 0, 2, and 3, with capabilities for
// reading datasets, groups, attributes, and various data layouts.
package hdf5

import (
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/scigolib/hdf5/internal/core"
	"github.com/scigolib/hdf5/internal/utils"
)

// File represents an open HDF5 file with its metadata and root group.
type File struct {
	osFile *os.File

	// reader is what every read goes through, and it is NOT always osFile. A file carrying a user block has its superblock at 512, 1024 or a further power-of-two offset, and every address inside the file is relative to that superblock rather than to byte zero. Wrapping the os.File in a SectionReader based there means the whole library below this point is unchanged: it keeps reading as though the superblock were at zero, because as far as it can see it is.
	reader utils.ReaderAt

	sb            *core.Superblock
	root          *Group
	visitedBTrees map[uint64]bool // Track visited B-tree addresses to prevent cycles
}

// Open opens an HDF5 file for reading and returns a File handle.
// The file must be a valid HDF5 file with a supported format version.
func Open(filename string) (*File, error) {
	//nolint:gosec // G304: User-provided filename is intentional for HDF5 file library
	f, err := os.Open(filename)
	if err != nil {
		return nil, utils.WrapError("file open failed", err)
	}

	// Get file size for address validation, and to bound the user-block search below.
	fi, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, utils.WrapError("file stat failed", err)
	}
	fileSize := fi.Size()

	// The superblock is not necessarily at byte zero. The format specification says to locate it by searching byte offset 0, then 512, then each successive power-of-two multiple -- the space before it is the user block, which exists so an HDF5 file can be wrapped in another format or carry a descriptive header. A MATLAB v7.3 .mat file is exactly that case: it is an HDF5 file behind a 512-byte header, so a reader that only checks offset zero rejects every one of them as "not an HDF5 file".
	userBlock, ok := findSuperblock(f, fileSize)
	if !ok {
		_ = f.Close()
		return nil, errors.New("not an HDF5 file")
	}

	var reader utils.ReaderAt = f
	if userBlock > 0 {
		// Addresses inside the file are relative to the superblock, so offsetting here is what keeps every address computation below this point correct without changing any of them.
		reader = io.NewSectionReader(f, userBlock, fileSize-userBlock)
	}

	sb, err := core.ReadSuperblock(reader)
	if err != nil {
		_ = f.Close()
		return nil, utils.WrapError("superblock read failed", err)
	}

	file := &File{
		osFile:        f,
		reader:        reader,
		sb:            sb,
		visitedBTrees: make(map[uint64]bool),
	}

	// Validate root group address.
	//nolint:gosec // G115: File size is always positive, safe to convert int64 to uint64
	if sb.RootGroup >= uint64(fileSize) {
		_ = f.Close()
		return nil, fmt.Errorf("root group address %d beyond file size %d",
			sb.RootGroup, fileSize)
	}

	// For all versions, sb.RootGroup now contains the correct object header address.
	file.root, err = loadGroup(file, sb.RootGroup)
	if err != nil {
		_ = f.Close()
		return nil, utils.WrapError("root group load failed", err)
	}

	// Ensure root group always has name "/" (may be empty from object header)
	file.root.name = "/"

	return file, nil
}

// isHDF5File verifies HDF5 file signature.
// findSuperblock returns the byte offset of the HDF5 signature, searching the offsets the format specification permits: 0, 512, 1024, 2048 and so on, each twice the last.
//
// The bound is the file size rather than a fixed number of attempts, so a large file with an unusually big user block is still found and a small one stops immediately. A signature at offset 0 is the overwhelmingly common case and costs one read.
func findSuperblock(r utils.ReaderAt, fileSize int64) (offset int64, ok bool) {
	buf := utils.GetBuffer(8)
	defer utils.ReleaseBuffer(buf)

	for off := int64(0); off+8 <= fileSize; {
		if _, err := r.ReadAt(buf, off); err == nil && string(buf) == core.Signature {
			return off, true
		}
		if off == 0 {
			off = 512
			continue
		}
		off *= 2
	}
	return 0, false
}

// Close closes the HDF5 file and releases associated resources.
// It is safe to call Close multiple times.
func (f *File) Close() error {
	if f.osFile == nil {
		return nil // Already closed.
	}
	err := f.osFile.Close()
	f.osFile = nil // Prevent double close.
	return err
}

// Root returns the root group of the HDF5 file.
func (f *File) Root() *Group {
	return f.root
}

// Walk traverses the entire file structure, calling fn for each object.
// Objects are visited in depth-first order starting from the root group.
func (f *File) Walk(fn func(path string, obj Object)) {
	walkGroup(f.root, "/", fn)
}

func walkGroup(g *Group, currentPath string, fn func(string, Object)) {
	fn(currentPath, g)

	for _, child := range g.Children() {
		childPath := currentPath + child.Name()

		if childGroup, ok := child.(*Group); ok {
			walkGroup(childGroup, childPath+"/", fn)
		} else {
			fn(childPath, child)
		}
	}
}

// SuperblockVersion returns the HDF5 superblock format version (0, 2, or 3).
func (f *File) SuperblockVersion() uint8 {
	return f.sb.Version
}

// Superblock returns the file's superblock metadata structure.
func (f *File) Superblock() *core.Superblock {
	return f.sb
}

// Reader returns the reader every address in this file is relative to, for low-level access.
//
// On a file with a user block this is NOT the os.File: it is a section of it based at the superblock, so an address read out of the file can be passed straight to it. Returning the raw os.File here would hand callers a reader whose offsets are wrong by the user block size on exactly the files where that is hardest to notice.
func (f *File) Reader() io.ReaderAt {
	return f.reader
}

// readSignature reads 4 bytes at address and returns string.
func readSignature(r io.ReaderAt, address uint64) string {
	buf := make([]byte, 4)
	//nolint:gosec // G115: HDF5 addresses fit in int64 for io.ReaderAt interface
	if _, err := r.ReadAt(buf, int64(address)); err != nil {
		return ""
	}
	return string(buf)
}
