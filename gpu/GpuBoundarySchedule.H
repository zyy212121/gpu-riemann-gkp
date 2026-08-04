#ifndef GpuBoundarySchedule_H
#define GpuBoundarySchedule_H

#include "IFstream.H"
#include "List.H"
#include "Tuple2.H"
#include "error.H"
#include "fileName.H"
#include "scalar.H"

#include <cmath>

namespace Foam
{

class GpuBoundaryScheduleTable
{
    List<Tuple2<scalar, scalar>> rows_;
    label cursor_;

    label indexAt(const scalar timeValue) const
    {
        if (timeValue < rows_.first().first())
        {
            return 0;
        }

        label low = 0;
        label high = rows_.size();
        while (low + 1 < high)
        {
            const label middle = low + (high - low)/2;
            if (rows_[middle].first() <= timeValue)
            {
                low = middle;
            }
            else
            {
                high = middle;
            }
        }
        return low;
    }

    void requireValid(const char* context) const
    {
        if (!validRows(rows_))
        {
            FatalErrorInFunction
                << context << ": schedule requires at least one finite row, "
                << "strictly increasing non-negative times, and finite values"
                << exit(FatalError);
        }
    }

public:
    explicit GpuBoundaryScheduleTable
    (
        const List<Tuple2<scalar, scalar>>& rows
    )
    :
        rows_(rows),
        cursor_(0)
    {
        requireValid("in-memory GPU boundary schedule");
    }

    explicit GpuBoundaryScheduleTable(const fileName& path)
    :
        rows_(),
        cursor_(0)
    {
        IFstream input(path);
        if (!input.good())
        {
            FatalErrorInFunction
                << "cannot read GPU boundary schedule " << path
                << exit(FatalError);
        }
        input >> rows_;
        requireValid(path.c_str());
    }

    static bool validRows(const List<Tuple2<scalar, scalar>>& rows)
    {
        if (rows.empty())
        {
            return false;
        }

        scalar previousTime = -1;
        forAll(rows, i)
        {
            const scalar timeValue = rows[i].first();
            const scalar value = rows[i].second();
            if
            (
                !std::isfinite(timeValue)
             || !std::isfinite(value)
             || timeValue < 0
             || (i > 0 && timeValue <= previousTime)
            )
            {
                return false;
            }
            previousTime = timeValue;
        }
        return true;
    }

    scalar valueAt(const scalar timeValue) const
    {
        return rows_[indexAt(timeValue)].second();
    }

    scalar linearValueAt(const scalar timeValue) const
    {
        if (timeValue <= rows_.first().first())
        {
            return rows_.first().second();
        }

        const label low = indexAt(timeValue);
        if (low + 1 >= rows_.size())
        {
            return rows_.last().second();
        }

        const scalar t0 = rows_[low].first();
        const scalar t1 = rows_[low + 1].first();
        const scalar fraction =
            min(max((timeValue - t0)/(t1 - t0), scalar(0)), scalar(1));
        return
            rows_[low].second()
          + fraction*(rows_[low + 1].second() - rows_[low].second());
    }

    void resetCursor(const scalar timeValue)
    {
        cursor_ = indexAt(timeValue);
    }

    bool advanceTo(const scalar timeValue)
    {
        const label next = indexAt(timeValue);
        const bool changed = next != cursor_;
        cursor_ = next;
        return changed;
    }

    label eventIndex() const
    {
        return cursor_;
    }

    scalar currentValue() const
    {
        return rows_[cursor_].second();
    }

    scalar nextEventTime() const
    {
        return cursor_ + 1 < rows_.size()
            ? rows_[cursor_ + 1].first()
            : GREAT;
    }

    label size() const
    {
        return rows_.size();
    }
};

} // End namespace Foam

#endif
