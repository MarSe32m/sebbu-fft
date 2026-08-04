// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

internal extension FixedWidthInteger {
    /// The smallest representable power of two greater than or equal to this value.
    ///
    /// Zero maps to one. Callers must supply a nonnegative value whose next power of
    /// two is representable by this integer type.
    @inlinable
    var nextPowerOfTwo: Self {
        guard self != 0 else { return 1 }
        return 1 << (Self.bitWidth - (self - 1).leadingZeroBitCount)
    }

    /// The base-two logarithm of this value, rounded down to an integer.
    ///
    /// - Precondition: The value is greater than zero.
    @inlinable
    var log2: Int {
        precondition(self > 0)
        return bitWidth - leadingZeroBitCount - 1
    }
}