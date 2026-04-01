#include "slang-perfect-hash.h"

#include "../core/slang-string-util.h"
#include "../core/slang-writer.h"

namespace Slang
{

// Implemented according to "Hash, displace, and compress"
// https://cmph.sourceforge.net/papers/esa09.pdf
HashFindResult minimalPerfectHash(const List<String>& ss, HashParams& hashParams)
{
    // Check for uniqueness
    for (Index i = 0; i < ss.getCount(); ++i)
    {
        for (Index j = i + 1; j < ss.getCount(); ++j)
        {
            if (ss[i] == ss[j])
            {
                return HashFindResult::NonUniqueKeys;
            }
        }
    }

    SLANG_ASSERT(UIndex(ss.getCount()) < std::numeric_limits<UInt32>::max());
    const UInt32 nBuckets = UInt32(ss.getCount());
    List<List<String>> initialBuckets;
    initialBuckets.setCount(nBuckets);

    const auto hash = [&](const String& s, const HashCode32 salt = 0) -> UInt32
    {
        //
        // The current getStableHashCode is susceptible to patterns of
        // collisions causing the search to fail for the SPIR-V opnames; it
        // performs poorly on short strings, taking over 300000 iterations to
        // diverge on "Ceil" and "FMix" (and place them in already unoccupied
        // slots)!
        //
        // Use FNV Hash here which seem perform much better on these short inputs
        // https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
        //
        // If you change this, don't forget to also sync the version below in
        // the printing code.
        UInt32 h = salt;
        for (const char c : s)
            h = (h * 0x01000193) ^ c;
        return h % nBuckets;
    };

    // Assign the inputs into their buckets according to the hash without salt.
    // Sort the buckets according to size, so that later we can make these have
    // unique destinations starting with the largest ones first as they are at
    // most risk of collision.
    for (const auto& s : ss)
    {
        initialBuckets[hash(s)].add(s);
    }
    initialBuckets.stableSort([](const List<String>& a, const List<String>& b)
                              { return a.getCount() > b.getCount(); });

    // These are our outputs, the salts are calculated such that for all input
    // word, x, hash(x, salt[hash(x, 0)]) is unique
    //
    // We keep the final table as we need to detect when we've been given a
    // word not in our language.
    hashParams.saltTable.setCount(nBuckets);
    for (auto& s : hashParams.saltTable)
        s = 0;
    hashParams.destTable.setCount(nBuckets);
    for (auto& s : hashParams.destTable)
        s.reduceLength(0);

    // This mask will, in each salt tryout, be used to prevent collisions
    // within a single bucket.
    List<bool> bucketDestinations = List<bool>::makeRepeated(false, nBuckets);

    for (const auto& b : initialBuckets)
    {
        // Break if we've reached the empty buckets
        if (!b.getCount())
        {
            break;
        }

        // Try out all the salts until we get one which has no internal
        // collisions for this bucket and also no collisions with the buckets
        // we've processed so far.
        UInt32 salt = 1;
        while (true)
        {
            bool collision = false;
            for (auto& d : bucketDestinations)
            {
                d = false;
            }

            for (const auto& s : b)
            {
                const auto i = hash(s, salt);
                if (hashParams.destTable[i].getLength() || bucketDestinations[i])
                {
                    collision = true;
                    break;
                }
                bucketDestinations[i] = true;
            }
            if (!collision)
            {
                break;
            }
            salt++;

            // If we fail to find a solution after some massive amount of tries
            // it's almost certainly because of some property of the hash
            // function and language causing an irresolvable collision.
            if (salt > 10000 * nBuckets)
            {
                return HashFindResult::UnavoidableHashCollision;
            }
        }
        for (const auto& s : b)
        {
            hashParams.saltTable[hash(s)] = salt;
            hashParams.destTable[hash(s, salt)] = s;
        }
    }
    return HashFindResult::Success;
}

String perfectHashToEmbeddableCpp(
    const HashParams& hashParams,
    const UnownedStringSlice& valueType,
    const UnownedStringSlice& funcName,
    const List<String>& values)
{
    SLANG_ASSERT(hashParams.saltTable.getCount() == hashParams.destTable.getCount());
    SLANG_ASSERT(hashParams.saltTable.getCount() == values.getCount());

    StringBuilder sb;
    StringWriter writer(&sb, WriterFlags(0));
    WriterHelper w(&writer);
    const auto line = [&](const char* l)
    {
        w.put(l);
        w.put("\n");
    };

    w.print(
        "bool %s(const UnownedStringSlice& str, %s& value)\n",
        String(funcName).getBuffer(),
        String(valueType).getBuffer());
    line("{");

    w.print("    static const unsigned tableSalt[%d] = {\n", (int)hashParams.saltTable.getCount());
    w.print("       ");
    for (Index i = 0; i < hashParams.saltTable.getCount(); ++i)
    {
        const auto salt = hashParams.saltTable[i];
        if (i != hashParams.saltTable.getCount() - 1)
        {
            w.print(" %d,", salt);
            if (i % 16 == 15)
            {
                w.print("\n       ");
            }
        }
        else
        {
            w.print(" %d", salt);
        }
    }
    line("\n    };");
    line("");

    w.print("    using KV = std::pair<const char*, %s>;\n", String(valueType).getBuffer());
    line("");

    w.print("    static const KV words[%d] =\n", (int)hashParams.destTable.getCount());
    line("    {");
    for (Index i = 0; i < hashParams.destTable.getCount(); ++i)
    {
        const auto& s = hashParams.destTable[i];
        const auto& v = values[i];
        w.print("        {\"%s\", %s},\n", s.getBuffer(), v.getBuffer());
    }
    line("    };");
    line("");

    // Make sure to update the hash function in the search function above if
    // you change this.
    line("    static const auto hash = [](const UnownedStringSlice& str, UInt32 salt){");
    line("        UInt32 h = salt;");
    line("        for (const char c : str)");
    line("            h = (h * 0x01000193) ^ c;");
    w.print("        return h %% %d;\n", (int)hashParams.saltTable.getCount());
    line("    };");
    line("");

    line("    const auto i = hash(str, tableSalt[hash(str, 0)]);");
    line("    if(str == words[i].first)");
    line("    {");
    line("        value = words[i].second;");
    line("        return true;");
    line("    }");
    line("    else");
    line("    {");
    line("        return false;");
    line("    }");
    line("}");
    line("");

    return sb.produceString();
}

} // namespace Slang
