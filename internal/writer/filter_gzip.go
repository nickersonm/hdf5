package writer

import (
	"bytes"
	"compress/zlib"
	"fmt"
	"io"
)

// HDF5 filter label for DEFLATE / GZIP compression. Extracted as a
// constant so goconst doesn't flag the duplicate string across source +
// helper tests.
const filterDeflateName = "deflate"

// GZIPFilter implements GZIP compression (FilterID = 1).
// This filter uses the DEFLATE compression algorithm to reduce data size.
// In HDF5, this filter is named "deflate" following zlib terminology.
//
// Compression levels:
//
//	1 = fastest compression, larger files
//	6 = balanced (default)
//	9 = best compression, slower
type GZIPFilter struct {
	level int // Compression level (1-9)
}

// NewGZIPFilter creates a GZIP filter with the specified compression level.
//
// Valid levels:
//
//	1 = Fast compression, lower ratio
//	6 = Default (balanced)
//	9 = Best compression, slower
//
// Invalid levels are automatically adjusted to 6 (default).
func NewGZIPFilter(level int) *GZIPFilter {
	if level < 1 || level > 9 {
		level = 6 // Default compression level
	}
	return &GZIPFilter{level: level}
}

// ID returns the HDF5 filter identifier for GZIP.
func (f *GZIPFilter) ID() FilterID {
	return FilterGZIP
}

// Name returns the HDF5 filter name.
// HDF5 uses "deflate" (the underlying algorithm) rather than "gzip".
func (f *GZIPFilter) Name() string {
	return filterDeflateName
}

// Apply compresses data with the DEFLATE algorithm in the ZLIB container, which is what HDF5's
// filter 1 is defined to hold.
//
// The container matters and the two are not interchangeable. HDF5 filter 1 is implemented against
// zlib's compress2/uncompress, so a chunk begins with the two-byte zlib header and ends with an
// Adler-32 checksum. Go's compress/gzip writes the GZIP container instead -- an 0x1f 0x8b magic,
// a ten-byte header and a CRC-32 trailer -- and the reference library rejects a chunk framed that
// way with "filter returned failure during read". The filter's own name, "deflate", is the hint:
// it names the algorithm, not the GZIP file format built on top of it.
func (f *GZIPFilter) Apply(data []byte) ([]byte, error) {
	var buf bytes.Buffer

	// Create zlib writer with specified compression level
	w, err := zlib.NewWriterLevel(&buf, f.level)
	if err != nil {
		return nil, fmt.Errorf("zlib writer creation failed: %w", err)
	}

	// Compress data
	if _, err := w.Write(data); err != nil {
		_ = w.Close() // Ignore close error on write failure
		return nil, fmt.Errorf("zlib compression failed: %w", err)
	}

	// Flush and close to ensure all data is written
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("zlib close failed: %w", err)
	}

	return buf.Bytes(), nil
}

// Remove decompresses ZLIB-framed DEFLATE data.
// Returns the original uncompressed data.
//
// This method reverses the Apply operation, restoring the original data.
func (f *GZIPFilter) Remove(data []byte) ([]byte, error) {
	buf := bytes.NewReader(data)

	// Create zlib reader
	r, err := zlib.NewReader(buf)
	if err != nil {
		return nil, fmt.Errorf("zlib reader creation failed: %w", err)
	}
	defer func() { _ = r.Close() }() // Ignore error in defer

	// Decompress data
	decompressed, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("zlib decompression failed: %w", err)
	}

	return decompressed, nil
}

// Encode returns the filter parameters for the Pipeline message.
//
// For GZIP, the client data contains a single value: the compression level.
// Flags are always 0 for GZIP.
func (f *GZIPFilter) Encode() (flags uint16, cdValues []uint32) {
	return 0, []uint32{uint32(f.level)} //nolint:gosec // G115: Compression level is 1-9, always fits in uint32
}
