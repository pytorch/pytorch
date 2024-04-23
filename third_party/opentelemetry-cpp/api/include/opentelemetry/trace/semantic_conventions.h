/*
 * Copyright The OpenTelemetry Authors
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  DO NOT EDIT, this is an Auto-generated file
  from buildscripts/semantic-convention/templates/SemanticAttributes.h.j2
*/

#pragma once

#include "opentelemetry/common/macros.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{

namespace SemanticConventions
{
/**
 * The URL of the OpenTelemetry schema for these keys and values.
 */
static constexpr const char *kSchemaUrl = "https://opentelemetry.io/schemas/1.24.0";

/**
 * The name of the invoked function.
 *
 * <p>Notes:
  <ul> <li>SHOULD be equal to the {@code faas.name} resource attribute of the invoked function.</li>
 </ul>
 */
static constexpr const char *kFaasInvokedName = "faas.invoked_name";

/**
 * The cloud provider of the invoked function.
 *
 * <p>Notes:
  <ul> <li>SHOULD be equal to the {@code cloud.provider} resource attribute of the invoked
 function.</li> </ul>
 */
static constexpr const char *kFaasInvokedProvider = "faas.invoked_provider";

/**
 * The cloud region of the invoked function.
 *
 * <p>Notes:
  <ul> <li>SHOULD be equal to the {@code cloud.region} resource attribute of the invoked
 function.</li> </ul>
 */
static constexpr const char *kFaasInvokedRegion = "faas.invoked_region";

/**
 * Type of the trigger which caused this function invocation.
 */
static constexpr const char *kFaasTrigger = "faas.trigger";

/**
 * The <a href="/docs/resource/README.md#service">{@code service.name}</a> of the remote service.
 * SHOULD be equal to the actual {@code service.name} resource attribute of the remote service if
 * any.
 */
static constexpr const char *kPeerService = "peer.service";

/**
 * Username or client_id extracted from the access token or <a
 * href="https://tools.ietf.org/html/rfc7235#section-4.2">Authorization</a> header in the inbound
 * request from outside the system.
 */
static constexpr const char *kEnduserId = "enduser.id";

/**
 * Actual/assumed role the client is making the request under extracted from token or application
 * security context.
 */
static constexpr const char *kEnduserRole = "enduser.role";

/**
 * Scopes or granted authorities the client currently possesses extracted from token or application
 * security context. The value would come from the scope associated with an <a
 * href="https://tools.ietf.org/html/rfc6749#section-3.3">OAuth 2.0 Access Token</a> or an attribute
 * value in a <a
 * href="http://docs.oasis-open.org/security/saml/Post2.0/sstc-saml-tech-overview-2.0.html">SAML 2.0
 * Assertion</a>.
 */
static constexpr const char *kEnduserScope = "enduser.scope";

/**
 * Identifies the class / type of event.
 *
 * <p>Notes:
  <ul> <li>Event names are subject to the same rules as <a
 href="https://github.com/open-telemetry/opentelemetry-specification/tree/v1.26.0/specification/common/attribute-naming.md">attribute
 names</a>. Notably, event names are namespaced to avoid collisions and provide a clean separation
 of semantics for events in separate domains like browser, mobile, and kubernetes.</li> </ul>
 */
static constexpr const char *kEventName = "event.name";

/**
 * A unique identifier for the Log Record.
 *
 * <p>Notes:
  <ul> <li>If an id is provided, other log records with the same id will be considered duplicates
and can be removed safely. This means, that two distinguishable log records MUST have different
values. The id MAY be an <a href="https://github.com/ulid/spec">Universally Unique Lexicographically
Sortable Identifier (ULID)</a>, but other identifiers (e.g. UUID) may be used as needed.</li> </ul>
 */
static constexpr const char *kLogRecordUid = "log.record.uid";

/**
 * The stream associated with the log. See below for a list of well-known values.
 */
static constexpr const char *kLogIostream = "log.iostream";

/**
 * The basename of the file.
 */
static constexpr const char *kLogFileName = "log.file.name";

/**
 * The basename of the file, with symlinks resolved.
 */
static constexpr const char *kLogFileNameResolved = "log.file.name_resolved";

/**
 * The full path to the file.
 */
static constexpr const char *kLogFilePath = "log.file.path";

/**
 * The full path to the file, with symlinks resolved.
 */
static constexpr const char *kLogFilePathResolved = "log.file.path_resolved";

/**
 * This attribute represents the state the application has transitioned into at the occurrence of
 the event.
 *
 * <p>Notes:
  <ul> <li>The iOS lifecycle states are defined in the <a
 href="https://developer.apple.com/documentation/uikit/uiapplicationdelegate#1656902">UIApplicationDelegate
 documentation</a>, and from which the {@code OS terminology} column values are derived.</li> </ul>
 */
static constexpr const char *kIosState = "ios.state";

/**
 * This attribute represents the state the application has transitioned into at the occurrence of
 the event.
 *
 * <p>Notes:
  <ul> <li>The Android lifecycle states are defined in <a
 href="https://developer.android.com/guide/components/activities/activity-lifecycle#lc">Activity
 lifecycle callbacks</a>, and from which the {@code OS identifiers} are derived.</li> </ul>
 */
static constexpr const char *kAndroidState = "android.state";

/**
 * The name of the connection pool; unique within the instrumented application. In case the
 * connection pool implementation doesn't provide a name, then the <a
 * href="/docs/database/database-spans.md#connection-level-attributes">db.connection_string</a>
 * should be used
 */
static constexpr const char *kPoolName = "pool.name";

/**
 * The state of a connection in the pool
 */
static constexpr const char *kState = "state";

/**
 * Full type name of the <a
 * href="https://learn.microsoft.com/dotnet/api/microsoft.aspnetcore.diagnostics.iexceptionhandler">{@code
 * IExceptionHandler}</a> implementation that handled the exception.
 */
static constexpr const char *kAspnetcoreDiagnosticsHandlerType =
    "aspnetcore.diagnostics.handler.type";

/**
 * Rate limiting policy name.
 */
static constexpr const char *kAspnetcoreRateLimitingPolicy = "aspnetcore.rate_limiting.policy";

/**
 * Rate-limiting result, shows whether the lease was acquired or contains a rejection reason
 */
static constexpr const char *kAspnetcoreRateLimitingResult = "aspnetcore.rate_limiting.result";

/**
 * Flag indicating if request was handled by the application pipeline.
 */
static constexpr const char *kAspnetcoreRequestIsUnhandled = "aspnetcore.request.is_unhandled";

/**
 * A value that indicates whether the matched route is a fallback route.
 */
static constexpr const char *kAspnetcoreRoutingIsFallback = "aspnetcore.routing.is_fallback";

/**
 * SignalR HTTP connection closure status.
 */
static constexpr const char *kSignalrConnectionStatus = "signalr.connection.status";

/**
 * <a
 * href="https://github.com/dotnet/aspnetcore/blob/main/src/SignalR/docs/specs/TransportProtocols.md">SignalR
 * transport type</a>
 */
static constexpr const char *kSignalrTransport = "signalr.transport";

/**
 * Name of the buffer pool.
 *
 * <p>Notes:
  <ul> <li>Pool names are generally obtained via <a
 href="https://docs.oracle.com/en/java/javase/11/docs/api/java.management/java/lang/management/BufferPoolMXBean.html#getName()">BufferPoolMXBean#getName()</a>.</li>
 </ul>
 */
static constexpr const char *kJvmBufferPoolName = "jvm.buffer.pool.name";

/**
 * Name of the memory pool.
 *
 * <p>Notes:
  <ul> <li>Pool names are generally obtained via <a
 href="https://docs.oracle.com/en/java/javase/11/docs/api/java.management/java/lang/management/MemoryPoolMXBean.html#getName()">MemoryPoolMXBean#getName()</a>.</li>
 </ul>
 */
static constexpr const char *kJvmMemoryPoolName = "jvm.memory.pool.name";

/**
 * The type of memory.
 */
static constexpr const char *kJvmMemoryType = "jvm.memory.type";

/**
 * The device identifier
 */
static constexpr const char *kSystemDevice = "system.device";

/**
 * The logical CPU number [0..n-1]
 */
static constexpr const char *kSystemCpuLogicalNumber = "system.cpu.logical_number";

/**
 * The state of the CPU
 */
static constexpr const char *kSystemCpuState = "system.cpu.state";

/**
 * The memory state
 */
static constexpr const char *kSystemMemoryState = "system.memory.state";

/**
 * The paging access direction
 */
static constexpr const char *kSystemPagingDirection = "system.paging.direction";

/**
 * The memory paging state
 */
static constexpr const char *kSystemPagingState = "system.paging.state";

/**
 * The memory paging type
 */
static constexpr const char *kSystemPagingType = "system.paging.type";

/**
 * The filesystem mode
 */
static constexpr const char *kSystemFilesystemMode = "system.filesystem.mode";

/**
 * The filesystem mount path
 */
static constexpr const char *kSystemFilesystemMountpoint = "system.filesystem.mountpoint";

/**
 * The filesystem state
 */
static constexpr const char *kSystemFilesystemState = "system.filesystem.state";

/**
 * The filesystem type
 */
static constexpr const char *kSystemFilesystemType = "system.filesystem.type";

/**
 * A stateless protocol MUST NOT set this attribute
 */
static constexpr const char *kSystemNetworkState = "system.network.state";

/**
 * The process state, e.g., <a
 * href="https://man7.org/linux/man-pages/man1/ps.1.html#PROCESS_STATE_CODES">Linux Process State
 * Codes</a>
 */
static constexpr const char *kSystemProcessesStatus = "system.processes.status";

/**
 * Client address - domain name if available without reverse DNS lookup; otherwise, IP address or
 Unix domain socket name.
 *
 * <p>Notes:
  <ul> <li>When observed from the server side, and when communicating through an intermediary,
 {@code client.address} SHOULD represent the client address behind any intermediaries,  for example
 proxies, if it's available.</li> </ul>
 */
static constexpr const char *kClientAddress = "client.address";

/**
 * Client port number.
 *
 * <p>Notes:
  <ul> <li>When observed from the server side, and when communicating through an intermediary,
 {@code client.port} SHOULD represent the client port behind any intermediaries,  for example
 proxies, if it's available.</li> </ul>
 */
static constexpr const char *kClientPort = "client.port";

/**
 * The column number in {@code code.filepath} best representing the operation. It SHOULD point
 * within the code unit named in {@code code.function}.
 */
static constexpr const char *kCodeColumn = "code.column";

/**
 * The source code file name that identifies the code unit as uniquely as possible (preferably an
 * absolute file path).
 */
static constexpr const char *kCodeFilepath = "code.filepath";

/**
 * The method or function name, or equivalent (usually rightmost part of the code unit's name).
 */
static constexpr const char *kCodeFunction = "code.function";

/**
 * The line number in {@code code.filepath} best representing the operation. It SHOULD point within
 * the code unit named in {@code code.function}.
 */
static constexpr const char *kCodeLineno = "code.lineno";

/**
 * The &quot;namespace&quot; within which {@code code.function} is defined. Usually the qualified
 * class or module name, such that {@code code.namespace} + some separator + {@code code.function}
 * form a unique identifier for the code unit.
 */
static constexpr const char *kCodeNamespace = "code.namespace";

/**
 * A stacktrace as a string in the natural representation for the language runtime. The
 * representation is to be determined and documented by each language SIG.
 */
static constexpr const char *kCodeStacktrace = "code.stacktrace";

/**
 * The consistency level of the query. Based on consistency values from <a
 * href="https://docs.datastax.com/en/cassandra-oss/3.0/cassandra/dml/dmlConfigConsistency.html">CQL</a>.
 */
static constexpr const char *kDbCassandraConsistencyLevel = "db.cassandra.consistency_level";

/**
 * The data center of the coordinating node for a query.
 */
static constexpr const char *kDbCassandraCoordinatorDc = "db.cassandra.coordinator.dc";

/**
 * The ID of the coordinating node for a query.
 */
static constexpr const char *kDbCassandraCoordinatorId = "db.cassandra.coordinator.id";

/**
 * Whether or not the query is idempotent.
 */
static constexpr const char *kDbCassandraIdempotence = "db.cassandra.idempotence";

/**
 * The fetch size used for paging, i.e. how many rows will be returned at once.
 */
static constexpr const char *kDbCassandraPageSize = "db.cassandra.page_size";

/**
 * The number of times a query was speculatively executed. Not set or {@code 0} if the query was not
 * executed speculatively.
 */
static constexpr const char *kDbCassandraSpeculativeExecutionCount =
    "db.cassandra.speculative_execution_count";

/**
 * The name of the primary Cassandra table that the operation is acting upon, including the keyspace
 name (if applicable).
 *
 * <p>Notes:
  <ul> <li>This mirrors the db.sql.table attribute but references cassandra rather than sql. It is
 not recommended to attempt any client-side parsing of {@code db.statement} just to get this
 property, but it should be set if it is provided by the library being instrumented. If the
 operation is acting upon an anonymous table, or more than one table, this value MUST NOT be
 set.</li> </ul>
 */
static constexpr const char *kDbCassandraTable = "db.cassandra.table";

/**
 * The connection string used to connect to the database. It is recommended to remove embedded
 * credentials.
 */
static constexpr const char *kDbConnectionString = "db.connection_string";

/**
 * Unique Cosmos client instance id.
 */
static constexpr const char *kDbCosmosdbClientId = "db.cosmosdb.client_id";

/**
 * Cosmos client connection mode.
 */
static constexpr const char *kDbCosmosdbConnectionMode = "db.cosmosdb.connection_mode";

/**
 * Cosmos DB container name.
 */
static constexpr const char *kDbCosmosdbContainer = "db.cosmosdb.container";

/**
 * CosmosDB Operation Type.
 */
static constexpr const char *kDbCosmosdbOperationType = "db.cosmosdb.operation_type";

/**
 * RU consumed for that operation
 */
static constexpr const char *kDbCosmosdbRequestCharge = "db.cosmosdb.request_charge";

/**
 * Request payload size in bytes
 */
static constexpr const char *kDbCosmosdbRequestContentLength = "db.cosmosdb.request_content_length";

/**
 * Cosmos DB status code.
 */
static constexpr const char *kDbCosmosdbStatusCode = "db.cosmosdb.status_code";

/**
 * Cosmos DB sub status code.
 */
static constexpr const char *kDbCosmosdbSubStatusCode = "db.cosmosdb.sub_status_code";

/**
 * Represents the identifier of an Elasticsearch cluster.
 */
static constexpr const char *kDbElasticsearchClusterName = "db.elasticsearch.cluster.name";

/**
 * Represents the human-readable identifier of the node/instance to which a request was routed.
 */
static constexpr const char *kDbElasticsearchNodeName = "db.elasticsearch.node.name";

/**
 * An identifier (address, unique name, or any other identifier) of the database instance that is
 * executing queries or mutations on the current connection. This is useful in cases where the
 * database is running in a clustered environment and the instrumentation is able to record the node
 * executing the query. The client may obtain this value in databases like MySQL using queries like
 * {@code select @@hostname}.
 */
static constexpr const char *kDbInstanceId = "db.instance.id";

/**
 * The fully-qualified class name of the <a
 * href="https://docs.oracle.com/javase/8/docs/technotes/guides/jdbc/">Java Database Connectivity
 * (JDBC)</a> driver used to connect.
 */
static constexpr const char *kDbJdbcDriverClassname = "db.jdbc.driver_classname";

/**
 * The MongoDB collection being accessed within the database stated in {@code db.name}.
 */
static constexpr const char *kDbMongodbCollection = "db.mongodb.collection";

/**
 * The Microsoft SQL Server <a
 href="https://docs.microsoft.com/sql/connect/jdbc/building-the-connection-url?view=sql-server-ver15">instance
 name</a> connecting to. This name is used to determine the port of a named instance.
 *
 * <p>Notes:
  <ul> <li>If setting a {@code db.mssql.instance_name}, {@code server.port} is no longer required
 (but still recommended if non-standard).</li> </ul>
 */
static constexpr const char *kDbMssqlInstanceName = "db.mssql.instance_name";

/**
 * This attribute is used to report the name of the database being accessed. For commands that
 switch the database, this should be set to the target database (even if the command fails).
 *
 * <p>Notes:
  <ul> <li>In some SQL databases, the database name to be used is called &quot;schema name&quot;. In
 case there are multiple layers that could be considered for database name (e.g. Oracle instance
 name and schema name), the database name to be used is the more specific layer (e.g. Oracle schema
 name).</li> </ul>
 */
static constexpr const char *kDbName = "db.name";

/**
 * The name of the operation being executed, e.g. the <a
 href="https://docs.mongodb.com/manual/reference/command/#database-operations">MongoDB command
 name</a> such as {@code findAndModify}, or the SQL keyword.
 *
 * <p>Notes:
  <ul> <li>When setting this to an SQL keyword, it is not recommended to attempt any client-side
 parsing of {@code db.statement} just to get this property, but it should be set if the operation
 name is provided by the library being instrumented. If the SQL statement has an ambiguous
 operation, or performs more than one operation, this value may be omitted.</li> </ul>
 */
static constexpr const char *kDbOperation = "db.operation";

/**
 * The index of the database being accessed as used in the <a
 * href="https://redis.io/commands/select">{@code SELECT} command</a>, provided as an integer. To be
 * used instead of the generic {@code db.name} attribute.
 */
static constexpr const char *kDbRedisDatabaseIndex = "db.redis.database_index";

/**
 * The name of the primary table that the operation is acting upon, including the database name (if
 applicable).
 *
 * <p>Notes:
  <ul> <li>It is not recommended to attempt any client-side parsing of {@code db.statement} just to
 get this property, but it should be set if it is provided by the library being instrumented. If the
 operation is acting upon an anonymous table, or more than one table, this value MUST NOT be
 set.</li> </ul>
 */
static constexpr const char *kDbSqlTable = "db.sql.table";

/**
 * The database statement being executed.
 */
static constexpr const char *kDbStatement = "db.statement";

/**
 * An identifier for the database management system (DBMS) product being used. See below for a list
 * of well-known identifiers.
 */
static constexpr const char *kDbSystem = "db.system";

/**
 * Username for accessing the database.
 */
static constexpr const char *kDbUser = "db.user";

/**
 * Deprecated, use {@code network.protocol.name} instead.
 *
 * @deprecated Deprecated, use `network.protocol.name` instead.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kHttpFlavor = "http.flavor";

/**
 * Deprecated, use {@code http.request.method} instead.
 *
 * @deprecated Deprecated, use `http.request.method` instead.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kHttpMethod = "http.method";

/**
 * Deprecated, use {@code http.request.header.content-length} instead.
 *
 * @deprecated Deprecated, use `http.request.header.content-length` instead.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kHttpRequestContentLength = "http.request_content_length";

/**
 * Deprecated, use {@code http.response.header.content-length} instead.
 *
 * @deprecated Deprecated, use `http.response.header.content-length` instead.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kHttpResponseContentLength = "http.response_content_length";

/**
 * Deprecated, use {@code url.scheme} instead.
 *
 * @deprecated Deprecated, use `url.scheme` instead.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kHttpScheme = "http.scheme";

/**
 * Deprecated, use {@code http.response.status_code} instead.
 *
 * @deprecated Deprecated, use `http.response.status_code` instead.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kHttpStatusCode = "http.status_code";

/**
 * Deprecated, use {@code url.path} and {@code url.query} instead.
 *
 * @deprecated Deprecated, use `url.path` and `url.query` instead.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kHttpTarget = "http.target";

/**
 * Deprecated, use {@code url.full} instead.
 *
 * @deprecated Deprecated, use `url.full` instead.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kHttpUrl = "http.url";

/**
 * Deprecated, use {@code user_agent.original} instead.
 *
 * @deprecated Deprecated, use `user_agent.original` instead.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kHttpUserAgent = "http.user_agent";

/**
 * Deprecated, use {@code server.address}.
 *
 * @deprecated Deprecated, use `server.address`.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kNetHostName = "net.host.name";

/**
 * Deprecated, use {@code server.port}.
 *
 * @deprecated Deprecated, use `server.port`.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kNetHostPort = "net.host.port";

/**
 * Deprecated, use {@code server.address} on client spans and {@code client.address} on server
 * spans.
 *
 * @deprecated Deprecated, use `server.address` on client spans and `client.address` on server
 * spans.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kNetPeerName = "net.peer.name";

/**
 * Deprecated, use {@code server.port} on client spans and {@code client.port} on server spans.
 *
 * @deprecated Deprecated, use `server.port` on client spans and `client.port` on server spans.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kNetPeerPort = "net.peer.port";

/**
 * Deprecated, use {@code network.protocol.name}.
 *
 * @deprecated Deprecated, use `network.protocol.name`.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kNetProtocolName = "net.protocol.name";

/**
 * Deprecated, use {@code network.protocol.version}.
 *
 * @deprecated Deprecated, use `network.protocol.version`.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kNetProtocolVersion = "net.protocol.version";

/**
 * Deprecated, use {@code network.transport} and {@code network.type}.
 *
 * @deprecated Deprecated, use `network.transport` and `network.type`.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kNetSockFamily = "net.sock.family";

/**
 * Deprecated, use {@code network.local.address}.
 *
 * @deprecated Deprecated, use `network.local.address`.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kNetSockHostAddr = "net.sock.host.addr";

/**
 * Deprecated, use {@code network.local.port}.
 *
 * @deprecated Deprecated, use `network.local.port`.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kNetSockHostPort = "net.sock.host.port";

/**
 * Deprecated, use {@code network.peer.address}.
 *
 * @deprecated Deprecated, use `network.peer.address`.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kNetSockPeerAddr = "net.sock.peer.addr";

/**
 * Deprecated, no replacement at this time.
 *
 * @deprecated Deprecated, no replacement at this time.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kNetSockPeerName = "net.sock.peer.name";

/**
 * Deprecated, use {@code network.peer.port}.
 *
 * @deprecated Deprecated, use `network.peer.port`.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kNetSockPeerPort = "net.sock.peer.port";

/**
 * Deprecated, use {@code network.transport}.
 *
 * @deprecated Deprecated, use `network.transport`.
 */
OPENTELEMETRY_DEPRECATED
static constexpr const char *kNetTransport = "net.transport";

/**
 * Destination address - domain name if available without reverse DNS lookup; otherwise, IP address
 or Unix domain socket name.
 *
 * <p>Notes:
  <ul> <li>When observed from the source side, and when communicating through an intermediary,
 {@code destination.address} SHOULD represent the destination address behind any intermediaries, for
 example proxies, if it's available.</li> </ul>
 */
static constexpr const char *kDestinationAddress = "destination.address";

/**
 * Destination port number
 */
static constexpr const char *kDestinationPort = "destination.port";

/**
 * The disk IO operation direction.
 */
static constexpr const char *kDiskIoDirection = "disk.io.direction";

/**
 * Describes a class of error the operation ended with.
 *
 * <p>Notes:
  <ul> <li>The {@code error.type} SHOULD be predictable and SHOULD have low cardinality.
Instrumentations SHOULD document the list of errors they report.</li><li>The cardinality of {@code
error.type} within one instrumentation library SHOULD be low. Telemetry consumers that aggregate
data from multiple instrumentation libraries and applications should be prepared for {@code
error.type} to have high cardinality at query time when no additional filters are
applied.</li><li>If the operation has completed successfully, instrumentations SHOULD NOT set {@code
error.type}.</li><li>If a specific domain defines its own set of error identifiers (such as HTTP or
gRPC status codes), it's RECOMMENDED to:</li><li>Use a domain-specific attribute</li> <li>Set {@code
error.type} to capture all errors, regardless of whether they are defined within the domain-specific
set or not.</li>
 </ul>
 */
static constexpr const char *kErrorType = "error.type";

/**
 * SHOULD be set to true if the exception event is recorded at a point where it is known that the
exception is escaping the scope of the span.
 *
 * <p>Notes:
  <ul> <li>An exception is considered to have escaped (or left) the scope of a span,
if that span is ended while the exception is still logically &quot;in flight&quot;.
This may be actually &quot;in flight&quot; in some languages (e.g. if the exception
is passed to a Context manager's {@code __exit__} method in Python) but will
usually be caught at the point of recording the exception in most languages.</li><li>It is usually
not possible to determine at the point where an exception is thrown whether it will escape the scope
of a span. However, it is trivial to know that an exception will escape, if one checks for an active
exception just before ending the span, as done in the <a href="#recording-an-exception">example for
recording span exceptions</a>.</li><li>It follows that an exception may still escape the scope of
the span even if the {@code exception.escaped} attribute was not set or set to false, since the
event might have been recorded at a time where it was not clear whether the exception will
escape.</li> </ul>
 */
static constexpr const char *kExceptionEscaped = "exception.escaped";

/**
 * The exception message.
 */
static constexpr const char *kExceptionMessage = "exception.message";

/**
 * A stacktrace as a string in the natural representation for the language runtime. The
 * representation is to be determined and documented by each language SIG.
 */
static constexpr const char *kExceptionStacktrace = "exception.stacktrace";

/**
 * The type of the exception (its fully-qualified class name, if applicable). The dynamic type of
 * the exception should be preferred over the static type in languages that support it.
 */
static constexpr const char *kExceptionType = "exception.type";

/**
 * The size of the request payload body in bytes. This is the number of bytes transferred excluding
 * headers and is often, but not always, present as the <a
 * href="https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length">Content-Length</a>
 * header. For requests using transport encoding, this should be the compressed size.
 */
static constexpr const char *kHttpRequestBodySize = "http.request.body.size";

/**
 * HTTP request method.
 *
 * <p>Notes:
  <ul> <li>HTTP request method value SHOULD be &quot;known&quot; to the instrumentation.
By default, this convention defines &quot;known&quot; methods as the ones listed in <a
href="https://www.rfc-editor.org/rfc/rfc9110.html#name-methods">RFC9110</a> and the PATCH method
defined in <a href="https://www.rfc-editor.org/rfc/rfc5789.html">RFC5789</a>.</li><li>If the HTTP
request method is not known to instrumentation, it MUST set the {@code http.request.method}
attribute to {@code _OTHER}.</li><li>If the HTTP instrumentation could end up converting valid HTTP
request methods to {@code _OTHER}, then it MUST provide a way to override the list of known HTTP
methods. If this override is done via environment variable, then the environment variable MUST be
named OTEL_INSTRUMENTATION_HTTP_KNOWN_METHODS and support a comma-separated list of case-sensitive
known HTTP methods (this list MUST be a full override of the default known method, it is not a list
of known methods in addition to the defaults).</li><li>HTTP method names are case-sensitive and
{@code http.request.method} attribute value MUST match a known HTTP method name exactly.
Instrumentations for specific web frameworks that consider HTTP methods to be case insensitive,
SHOULD populate a canonical equivalent. Tracing instrumentations that do so, MUST also set {@code
http.request.method_original} to the original value.</li> </ul>
 */
static constexpr const char *kHttpRequestMethod = "http.request.method";

/**
 * Original HTTP method sent by the client in the request line.
 */
static constexpr const char *kHttpRequestMethodOriginal = "http.request.method_original";

/**
 * The ordinal number of request resending attempt (for any reason, including redirects).
 *
 * <p>Notes:
  <ul> <li>The resend count SHOULD be updated each time an HTTP request gets resent by the client,
 regardless of what was the cause of the resending (e.g. redirection, authorization failure, 503
 Server Unavailable, network issues, or any other).</li> </ul>
 */
static constexpr const char *kHttpRequestResendCount = "http.request.resend_count";

/**
 * The size of the response payload body in bytes. This is the number of bytes transferred excluding
 * headers and is often, but not always, present as the <a
 * href="https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length">Content-Length</a>
 * header. For requests using transport encoding, this should be the compressed size.
 */
static constexpr const char *kHttpResponseBodySize = "http.response.body.size";

/**
 * <a href="https://tools.ietf.org/html/rfc7231#section-6">HTTP response status code</a>.
 */
static constexpr const char *kHttpResponseStatusCode = "http.response.status_code";

/**
 * The matched route, that is, the path template in the format used by the respective server
framework.
 *
 * <p>Notes:
  <ul> <li>MUST NOT be populated when this is not supported by the HTTP server framework as the
route attribute should have low-cardinality and the URI path can NOT substitute it. SHOULD include
the <a href="/docs/http/http-spans.md#http-server-definitions">application root</a> if there is
one.</li> </ul>
 */
static constexpr const char *kHttpRoute = "http.route";

/**
 * The number of messages sent, received, or processed in the scope of the batching operation.
 *
 * <p>Notes:
  <ul> <li>Instrumentations SHOULD NOT set {@code messaging.batch.message_count} on spans that
 operate with a single message. When a messaging client library supports both batch and
 single-message API for the same operation, instrumentations SHOULD use {@code
 messaging.batch.message_count} for batching APIs and SHOULD NOT use it for single-message
 APIs.</li> </ul>
 */
static constexpr const char *kMessagingBatchMessageCount = "messaging.batch.message_count";

/**
 * A unique identifier for the client that consumes or produces a message.
 */
static constexpr const char *kMessagingClientId = "messaging.client_id";

/**
 * A boolean that is true if the message destination is anonymous (could be unnamed or have
 * auto-generated name).
 */
static constexpr const char *kMessagingDestinationAnonymous = "messaging.destination.anonymous";

/**
 * The message destination name
 *
 * <p>Notes:
  <ul> <li>Destination name SHOULD uniquely identify a specific queue, topic or other entity within
the broker. If the broker doesn't have such notion, the destination name SHOULD uniquely identify
the broker.</li> </ul>
 */
static constexpr const char *kMessagingDestinationName = "messaging.destination.name";

/**
 * Low cardinality representation of the messaging destination name
 *
 * <p>Notes:
  <ul> <li>Destination names could be constructed from templates. An example would be a destination
 name involving a user name or product id. Although the destination name in this case is of high
 cardinality, the underlying template is of low cardinality and can be effectively used for grouping
 and aggregation.</li> </ul>
 */
static constexpr const char *kMessagingDestinationTemplate = "messaging.destination.template";

/**
 * A boolean that is true if the message destination is temporary and might not exist anymore after
 * messages are processed.
 */
static constexpr const char *kMessagingDestinationTemporary = "messaging.destination.temporary";

/**
 * A boolean that is true if the publish message destination is anonymous (could be unnamed or have
 * auto-generated name).
 */
static constexpr const char *kMessagingDestinationPublishAnonymous =
    "messaging.destination_publish.anonymous";

/**
 * The name of the original destination the message was published to
 *
 * <p>Notes:
  <ul> <li>The name SHOULD uniquely identify a specific queue, topic, or other entity within the
broker. If the broker doesn't have such notion, the original destination name SHOULD uniquely
identify the broker.</li> </ul>
 */
static constexpr const char *kMessagingDestinationPublishName =
    "messaging.destination_publish.name";

/**
 * The ordering key for a given message. If the attribute is not present, the message does not have
 * an ordering key.
 */
static constexpr const char *kMessagingGcpPubsubMessageOrderingKey =
    "messaging.gcp_pubsub.message.ordering_key";

/**
 * Name of the Kafka Consumer Group that is handling the message. Only applies to consumers, not
 * producers.
 */
static constexpr const char *kMessagingKafkaConsumerGroup = "messaging.kafka.consumer.group";

/**
 * Partition the message is sent to.
 */
static constexpr const char *kMessagingKafkaDestinationPartition =
    "messaging.kafka.destination.partition";

/**
 * Message keys in Kafka are used for grouping alike messages to ensure they're processed on the
 same partition. They differ from {@code messaging.message.id} in that they're not unique. If the
 key is {@code null}, the attribute MUST NOT be set.
 *
 * <p>Notes:
  <ul> <li>If the key type is not string, it's string representation has to be supplied for the
 attribute. If the key has no unambiguous, canonical string form, don't include its value.</li>
 </ul>
 */
static constexpr const char *kMessagingKafkaMessageKey = "messaging.kafka.message.key";

/**
 * The offset of a record in the corresponding Kafka partition.
 */
static constexpr const char *kMessagingKafkaMessageOffset = "messaging.kafka.message.offset";

/**
 * A boolean that is true if the message is a tombstone.
 */
static constexpr const char *kMessagingKafkaMessageTombstone = "messaging.kafka.message.tombstone";

/**
 * The size of the message body in bytes.
 *
 * <p>Notes:
  <ul> <li>This can refer to both the compressed or uncompressed body size. If both sizes are known,
the uncompressed body size should be used.</li> </ul>
 */
static constexpr const char *kMessagingMessageBodySize = "messaging.message.body.size";

/**
 * The conversation ID identifying the conversation to which the message belongs, represented as a
 * string. Sometimes called &quot;Correlation ID&quot;.
 */
static constexpr const char *kMessagingMessageConversationId = "messaging.message.conversation_id";

/**
 * The size of the message body and metadata in bytes.
 *
 * <p>Notes:
  <ul> <li>This can refer to both the compressed or uncompressed size. If both sizes are known, the
uncompressed size should be used.</li> </ul>
 */
static constexpr const char *kMessagingMessageEnvelopeSize = "messaging.message.envelope.size";

/**
 * A value used by the messaging system as an identifier for the message, represented as a string.
 */
static constexpr const char *kMessagingMessageId = "messaging.message.id";

/**
 * A string identifying the kind of messaging operation.
 *
 * <p>Notes:
  <ul> <li>If a custom value is used, it MUST be of low cardinality.</li> </ul>
 */
static constexpr const char *kMessagingOperation = "messaging.operation";

/**
 * RabbitMQ message routing key.
 */
static constexpr const char *kMessagingRabbitmqDestinationRoutingKey =
    "messaging.rabbitmq.destination.routing_key";

/**
 * Name of the RocketMQ producer/consumer group that is handling the message. The client type is
 * identified by the SpanKind.
 */
static constexpr const char *kMessagingRocketmqClientGroup = "messaging.rocketmq.client_group";

/**
 * Model of message consumption. This only applies to consumer spans.
 */
static constexpr const char *kMessagingRocketmqConsumptionModel =
    "messaging.rocketmq.consumption_model";

/**
 * The delay time level for delay message, which determines the message delay time.
 */
static constexpr const char *kMessagingRocketmqMessageDelayTimeLevel =
    "messaging.rocketmq.message.delay_time_level";

/**
 * The timestamp in milliseconds that the delay message is expected to be delivered to consumer.
 */
static constexpr const char *kMessagingRocketmqMessageDeliveryTimestamp =
    "messaging.rocketmq.message.delivery_timestamp";

/**
 * It is essential for FIFO message. Messages that belong to the same message group are always
 * processed one by one within the same consumer group.
 */
static constexpr const char *kMessagingRocketmqMessageGroup = "messaging.rocketmq.message.group";

/**
 * Key(s) of message, another way to mark message besides message id.
 */
static constexpr const char *kMessagingRocketmqMessageKeys = "messaging.rocketmq.message.keys";

/**
 * The secondary classifier of message besides topic.
 */
static constexpr const char *kMessagingRocketmqMessageTag = "messaging.rocketmq.message.tag";

/**
 * Type of message.
 */
static constexpr const char *kMessagingRocketmqMessageType = "messaging.rocketmq.message.type";

/**
 * Namespace of RocketMQ resources, resources in different namespaces are individual.
 */
static constexpr const char *kMessagingRocketmqNamespace = "messaging.rocketmq.namespace";

/**
 * An identifier for the messaging system being used. See below for a list of well-known
 * identifiers.
 */
static constexpr const char *kMessagingSystem = "messaging.system";

/**
 * The ISO 3166-1 alpha-2 2-character country code associated with the mobile carrier network.
 */
static constexpr const char *kNetworkCarrierIcc = "network.carrier.icc";

/**
 * The mobile carrier country code.
 */
static constexpr const char *kNetworkCarrierMcc = "network.carrier.mcc";

/**
 * The mobile carrier network code.
 */
static constexpr const char *kNetworkCarrierMnc = "network.carrier.mnc";

/**
 * The name of the mobile carrier.
 */
static constexpr const char *kNetworkCarrierName = "network.carrier.name";

/**
 * This describes more details regarding the connection.type. It may be the type of cell technology
 * connection, but it could be used for describing details about a wifi connection.
 */
static constexpr const char *kNetworkConnectionSubtype = "network.connection.subtype";

/**
 * The internet connection type.
 */
static constexpr const char *kNetworkConnectionType = "network.connection.type";

/**
 * The network IO operation direction.
 */
static constexpr const char *kNetworkIoDirection = "network.io.direction";

/**
 * Local address of the network connection - IP address or Unix domain socket name.
 */
static constexpr const char *kNetworkLocalAddress = "network.local.address";

/**
 * Local port number of the network connection.
 */
static constexpr const char *kNetworkLocalPort = "network.local.port";

/**
 * Peer address of the network connection - IP address or Unix domain socket name.
 */
static constexpr const char *kNetworkPeerAddress = "network.peer.address";

/**
 * Peer port number of the network connection.
 */
static constexpr const char *kNetworkPeerPort = "network.peer.port";

/**
 * <a href="https://osi-model.com/application-layer/">OSI application layer</a> or non-OSI
 equivalent.
 *
 * <p>Notes:
  <ul> <li>The value SHOULD be normalized to lowercase.</li> </ul>
 */
static constexpr const char *kNetworkProtocolName = "network.protocol.name";

/**
 * Version of the protocol specified in {@code network.protocol.name}.
 *
 * <p>Notes:
  <ul> <li>{@code network.protocol.version} refers to the version of the protocol used and might be
 different from the protocol client's version. If the HTTP client has a version of {@code 0.27.2},
 but sends HTTP version {@code 1.1}, this attribute should be set to {@code 1.1}.</li> </ul>
 */
static constexpr const char *kNetworkProtocolVersion = "network.protocol.version";

/**
 * <a href="https://osi-model.com/transport-layer/">OSI transport layer</a> or <a
href="https://wikipedia.org/wiki/Inter-process_communication">inter-process communication
method</a>.
 *
 * <p>Notes:
  <ul> <li>The value SHOULD be normalized to lowercase.</li><li>Consider always setting the
transport when setting a port number, since a port number is ambiguous without knowing the
transport. For example different processes could be listening on TCP port 12345 and UDP port
12345.</li> </ul>
 */
static constexpr const char *kNetworkTransport = "network.transport";

/**
 * <a href="https://osi-model.com/network-layer/">OSI network layer</a> or non-OSI equivalent.
 *
 * <p>Notes:
  <ul> <li>The value SHOULD be normalized to lowercase.</li> </ul>
 */
static constexpr const char *kNetworkType = "network.type";

/**
 * The <a href="https://connect.build/docs/protocol/#error-codes">error codes</a> of the Connect
 * request. Error codes are always string values.
 */
static constexpr const char *kRpcConnectRpcErrorCode = "rpc.connect_rpc.error_code";

/**
 * The <a href="https://github.com/grpc/grpc/blob/v1.33.2/doc/statuscodes.md">numeric status
 * code</a> of the gRPC request.
 */
static constexpr const char *kRpcGrpcStatusCode = "rpc.grpc.status_code";

/**
 * {@code error.code} property of response if it is an error response.
 */
static constexpr const char *kRpcJsonrpcErrorCode = "rpc.jsonrpc.error_code";

/**
 * {@code error.message} property of response if it is an error response.
 */
static constexpr const char *kRpcJsonrpcErrorMessage = "rpc.jsonrpc.error_message";

/**
 * {@code id} property of request or response. Since protocol allows id to be int, string, {@code
 * null} or missing (for notifications), value is expected to be cast to string for simplicity. Use
 * empty string in case of {@code null} value. Omit entirely if this is a notification.
 */
static constexpr const char *kRpcJsonrpcRequestId = "rpc.jsonrpc.request_id";

/**
 * Protocol version as in {@code jsonrpc} property of request/response. Since JSON-RPC 1.0 doesn't
 * specify this, the value can be omitted.
 */
static constexpr const char *kRpcJsonrpcVersion = "rpc.jsonrpc.version";

/**
 * The name of the (logical) method being called, must be equal to the $method part in the span
 name.
 *
 * <p>Notes:
  <ul> <li>This is the logical name of the method from the RPC interface perspective, which can be
 different from the name of any implementing method/function. The {@code code.function} attribute
 may be used to store the latter (e.g., method actually executing the call on the server side, RPC
 client stub method on the client side).</li> </ul>
 */
static constexpr const char *kRpcMethod = "rpc.method";

/**
 * The full (logical) name of the service being called, including its package name, if applicable.
 *
 * <p>Notes:
  <ul> <li>This is the logical name of the service from the RPC interface perspective, which can be
 different from the name of any implementing class. The {@code code.namespace} attribute may be used
 to store the latter (despite the attribute name, it may include a class name; e.g., class with
 method actually executing the call on the server side, RPC client stub class on the client
 side).</li> </ul>
 */
static constexpr const char *kRpcService = "rpc.service";

/**
 * A string identifying the remoting system. See below for a list of well-known identifiers.
 */
static constexpr const char *kRpcSystem = "rpc.system";

/**
 * Server domain name if available without reverse DNS lookup; otherwise, IP address or Unix domain
 socket name.
 *
 * <p>Notes:
  <ul> <li>When observed from the client side, and when communicating through an intermediary,
 {@code server.address} SHOULD represent the server address behind any intermediaries, for example
 proxies, if it's available.</li> </ul>
 */
static constexpr const char *kServerAddress = "server.address";

/**
 * Server port number.
 *
 * <p>Notes:
  <ul> <li>When observed from the client side, and when communicating through an intermediary,
 {@code server.port} SHOULD represent the server port behind any intermediaries, for example
 proxies, if it's available.</li> </ul>
 */
static constexpr const char *kServerPort = "server.port";

/**
 * Source address - domain name if available without reverse DNS lookup; otherwise, IP address or
 Unix domain socket name.
 *
 * <p>Notes:
  <ul> <li>When observed from the destination side, and when communicating through an intermediary,
 {@code source.address} SHOULD represent the source address behind any intermediaries, for example
 proxies, if it's available.</li> </ul>
 */
static constexpr const char *kSourceAddress = "source.address";

/**
 * Source port number
 */
static constexpr const char *kSourcePort = "source.port";

/**
 * Current &quot;managed&quot; thread ID (as opposed to OS thread ID).
 */
static constexpr const char *kThreadId = "thread.id";

/**
 * Current thread name.
 */
static constexpr const char *kThreadName = "thread.name";

/**
 * String indicating the <a
 href="https://datatracker.ietf.org/doc/html/rfc5246#appendix-A.5">cipher</a> used during the
 current connection.
 *
 * <p>Notes:
  <ul> <li>The values allowed for {@code tls.cipher} MUST be one of the {@code Descriptions} of the
 <a
 href="https://www.iana.org/assignments/tls-parameters/tls-parameters.xhtml#table-tls-parameters-4">registered
 TLS Cipher Suits</a>.</li> </ul>
 */
static constexpr const char *kTlsCipher = "tls.cipher";

/**
 * PEM-encoded stand-alone certificate offered by the client. This is usually mutually-exclusive of
 * {@code client.certificate_chain} since this value also exists in that list.
 */
static constexpr const char *kTlsClientCertificate = "tls.client.certificate";

/**
 * Array of PEM-encoded certificates that make up the certificate chain offered by the client. This
 * is usually mutually-exclusive of {@code client.certificate} since that value should be the first
 * certificate in the chain.
 */
static constexpr const char *kTlsClientCertificateChain = "tls.client.certificate_chain";

/**
 * Certificate fingerprint using the MD5 digest of DER-encoded version of certificate offered by the
 * client. For consistency with other hash values, this value should be formatted as an uppercase
 * hash.
 */
static constexpr const char *kTlsClientHashMd5 = "tls.client.hash.md5";

/**
 * Certificate fingerprint using the SHA1 digest of DER-encoded version of certificate offered by
 * the client. For consistency with other hash values, this value should be formatted as an
 * uppercase hash.
 */
static constexpr const char *kTlsClientHashSha1 = "tls.client.hash.sha1";

/**
 * Certificate fingerprint using the SHA256 digest of DER-encoded version of certificate offered by
 * the client. For consistency with other hash values, this value should be formatted as an
 * uppercase hash.
 */
static constexpr const char *kTlsClientHashSha256 = "tls.client.hash.sha256";

/**
 * Distinguished name of <a
 * href="https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.6">subject</a> of the issuer of
 * the x.509 certificate presented by the client.
 */
static constexpr const char *kTlsClientIssuer = "tls.client.issuer";

/**
 * A hash that identifies clients based on how they perform an SSL/TLS handshake.
 */
static constexpr const char *kTlsClientJa3 = "tls.client.ja3";

/**
 * Date/Time indicating when client certificate is no longer considered valid.
 */
static constexpr const char *kTlsClientNotAfter = "tls.client.not_after";

/**
 * Date/Time indicating when client certificate is first considered valid.
 */
static constexpr const char *kTlsClientNotBefore = "tls.client.not_before";

/**
 * Also called an SNI, this tells the server which hostname to which the client is attempting to
 * connect to.
 */
static constexpr const char *kTlsClientServerName = "tls.client.server_name";

/**
 * Distinguished name of subject of the x.509 certificate presented by the client.
 */
static constexpr const char *kTlsClientSubject = "tls.client.subject";

/**
 * Array of ciphers offered by the client during the client hello.
 */
static constexpr const char *kTlsClientSupportedCiphers = "tls.client.supported_ciphers";

/**
 * String indicating the curve used for the given cipher, when applicable
 */
static constexpr const char *kTlsCurve = "tls.curve";

/**
 * Boolean flag indicating if the TLS negotiation was successful and transitioned to an encrypted
 * tunnel.
 */
static constexpr const char *kTlsEstablished = "tls.established";

/**
 * String indicating the protocol being tunneled. Per the values in the <a
 * href="https://www.iana.org/assignments/tls-extensiontype-values/tls-extensiontype-values.xhtml#alpn-protocol-ids">IANA
 * registry</a>, this string should be lower case.
 */
static constexpr const char *kTlsNextProtocol = "tls.next_protocol";

/**
 * Normalized lowercase protocol name parsed from original string of the negotiated <a
 * href="https://www.openssl.org/docs/man1.1.1/man3/SSL_get_version.html#RETURN-VALUES">SSL/TLS
 * protocol version</a>
 */
static constexpr const char *kTlsProtocolName = "tls.protocol.name";

/**
 * Numeric part of the version parsed from the original string of the negotiated <a
 * href="https://www.openssl.org/docs/man1.1.1/man3/SSL_get_version.html#RETURN-VALUES">SSL/TLS
 * protocol version</a>
 */
static constexpr const char *kTlsProtocolVersion = "tls.protocol.version";

/**
 * Boolean flag indicating if this TLS connection was resumed from an existing TLS negotiation.
 */
static constexpr const char *kTlsResumed = "tls.resumed";

/**
 * PEM-encoded stand-alone certificate offered by the server. This is usually mutually-exclusive of
 * {@code server.certificate_chain} since this value also exists in that list.
 */
static constexpr const char *kTlsServerCertificate = "tls.server.certificate";

/**
 * Array of PEM-encoded certificates that make up the certificate chain offered by the server. This
 * is usually mutually-exclusive of {@code server.certificate} since that value should be the first
 * certificate in the chain.
 */
static constexpr const char *kTlsServerCertificateChain = "tls.server.certificate_chain";

/**
 * Certificate fingerprint using the MD5 digest of DER-encoded version of certificate offered by the
 * server. For consistency with other hash values, this value should be formatted as an uppercase
 * hash.
 */
static constexpr const char *kTlsServerHashMd5 = "tls.server.hash.md5";

/**
 * Certificate fingerprint using the SHA1 digest of DER-encoded version of certificate offered by
 * the server. For consistency with other hash values, this value should be formatted as an
 * uppercase hash.
 */
static constexpr const char *kTlsServerHashSha1 = "tls.server.hash.sha1";

/**
 * Certificate fingerprint using the SHA256 digest of DER-encoded version of certificate offered by
 * the server. For consistency with other hash values, this value should be formatted as an
 * uppercase hash.
 */
static constexpr const char *kTlsServerHashSha256 = "tls.server.hash.sha256";

/**
 * Distinguished name of <a
 * href="https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.6">subject</a> of the issuer of
 * the x.509 certificate presented by the client.
 */
static constexpr const char *kTlsServerIssuer = "tls.server.issuer";

/**
 * A hash that identifies servers based on how they perform an SSL/TLS handshake.
 */
static constexpr const char *kTlsServerJa3s = "tls.server.ja3s";

/**
 * Date/Time indicating when server certificate is no longer considered valid.
 */
static constexpr const char *kTlsServerNotAfter = "tls.server.not_after";

/**
 * Date/Time indicating when server certificate is first considered valid.
 */
static constexpr const char *kTlsServerNotBefore = "tls.server.not_before";

/**
 * Distinguished name of subject of the x.509 certificate presented by the server.
 */
static constexpr const char *kTlsServerSubject = "tls.server.subject";

/**
 * The <a href="https://www.rfc-editor.org/rfc/rfc3986#section-3.5">URI fragment</a> component
 */
static constexpr const char *kUrlFragment = "url.fragment";

/**
 * Absolute URL describing a network resource according to <a
href="https://www.rfc-editor.org/rfc/rfc3986">RFC3986</a>
 *
 * <p>Notes:
  <ul> <li>For network calls, URL usually has {@code scheme://host[:port][path][?query][#fragment]}
format, where the fragment is not transmitted over HTTP, but if it is known, it SHOULD be included
nevertheless.
{@code url.full} MUST NOT contain credentials passed via URL in form of {@code
https://username:password@www.example.com/}. In such case username and password SHOULD be redacted
and attribute's value SHOULD be {@code https://REDACTED:REDACTED@www.example.com/}.
{@code url.full} SHOULD capture the absolute URL when it is available (or can be reconstructed) and
SHOULD NOT be validated or modified except for sanitizing purposes.</li> </ul>
 */
static constexpr const char *kUrlFull = "url.full";

/**
 * The <a href="https://www.rfc-editor.org/rfc/rfc3986#section-3.3">URI path</a> component
 */
static constexpr const char *kUrlPath = "url.path";

/**
 * The <a href="https://www.rfc-editor.org/rfc/rfc3986#section-3.4">URI query</a> component
 *
 * <p>Notes:
  <ul> <li>Sensitive content provided in query string SHOULD be scrubbed when instrumentations can
 identify it.</li> </ul>
 */
static constexpr const char *kUrlQuery = "url.query";

/**
 * The <a href="https://www.rfc-editor.org/rfc/rfc3986#section-3.1">URI scheme</a> component
 * identifying the used protocol.
 */
static constexpr const char *kUrlScheme = "url.scheme";

/**
 * Value of the <a href="https://www.rfc-editor.org/rfc/rfc9110.html#field.user-agent">HTTP
 * User-Agent</a> header sent by the client.
 */
static constexpr const char *kUserAgentOriginal = "user_agent.original";

/**
 * A unique id to identify a session.
 */
static constexpr const char *kSessionId = "session.id";

/**
 * The previous {@code session.id} for this user, when known.
 */
static constexpr const char *kSessionPreviousId = "session.previous_id";

/**
 * The full invoked ARN as provided on the {@code Context} passed to the function ({@code
 Lambda-Runtime-Invoked-Function-Arn} header on the {@code /runtime/invocation/next} applicable).
 *
 * <p>Notes:
  <ul> <li>This may be different from {@code cloud.resource_id} if an alias is involved.</li> </ul>
 */
static constexpr const char *kAwsLambdaInvokedArn = "aws.lambda.invoked_arn";

/**
 * The <a href="https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#id">event_id</a>
 * uniquely identifies the event.
 */
static constexpr const char *kCloudeventsEventId = "cloudevents.event_id";

/**
 * The <a
 * href="https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#source-1">source</a>
 * identifies the context in which an event happened.
 */
static constexpr const char *kCloudeventsEventSource = "cloudevents.event_source";

/**
 * The <a
 * href="https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#specversion">version of
 * the CloudEvents specification</a> which the event uses.
 */
static constexpr const char *kCloudeventsEventSpecVersion = "cloudevents.event_spec_version";

/**
 * The <a
 * href="https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#subject">subject</a> of
 * the event in the context of the event producer (identified by source).
 */
static constexpr const char *kCloudeventsEventSubject = "cloudevents.event_subject";

/**
 * The <a
 * href="https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#type">event_type</a>
 * contains a value describing the type of event related to the originating occurrence.
 */
static constexpr const char *kCloudeventsEventType = "cloudevents.event_type";

/**
 * Parent-child Reference type
 *
 * <p>Notes:
  <ul> <li>The causal relationship between a child Span and a parent Span.</li> </ul>
 */
static constexpr const char *kOpentracingRefType = "opentracing.ref_type";

/**
 * Name of the code, either &quot;OK&quot; or &quot;ERROR&quot;. MUST NOT be set if the status code
 * is UNSET.
 */
static constexpr const char *kOtelStatusCode = "otel.status_code";

/**
 * Description of the Status if it has a value, otherwise not set.
 */
static constexpr const char *kOtelStatusDescription = "otel.status_description";

/**
 * The invocation ID of the current function invocation.
 */
static constexpr const char *kFaasInvocationId = "faas.invocation_id";

/**
 * The name of the source on which the triggering operation was performed. For example, in Cloud
 * Storage or S3 corresponds to the bucket name, and in Cosmos DB to the database name.
 */
static constexpr const char *kFaasDocumentCollection = "faas.document.collection";

/**
 * The document name/table subjected to the operation. For example, in Cloud Storage or S3 is the
 * name of the file, and in Cosmos DB the table name.
 */
static constexpr const char *kFaasDocumentName = "faas.document.name";

/**
 * Describes the type of the operation that was performed on the data.
 */
static constexpr const char *kFaasDocumentOperation = "faas.document.operation";

/**
 * A string containing the time when the data was accessed in the <a
 * href="https://www.iso.org/iso-8601-date-and-time-format.html">ISO 8601</a> format expressed in <a
 * href="https://www.w3.org/TR/NOTE-datetime">UTC</a>.
 */
static constexpr const char *kFaasDocumentTime = "faas.document.time";

/**
 * A string containing the schedule period as <a
 * href="https://docs.oracle.com/cd/E12058_01/doc/doc.1014/e12030/cron_expressions.htm">Cron
 * Expression</a>.
 */
static constexpr const char *kFaasCron = "faas.cron";

/**
 * A string containing the function invocation time in the <a
 * href="https://www.iso.org/iso-8601-date-and-time-format.html">ISO 8601</a> format expressed in <a
 * href="https://www.w3.org/TR/NOTE-datetime">UTC</a>.
 */
static constexpr const char *kFaasTime = "faas.time";

/**
 * A boolean that is true if the serverless function is executed for the first time (aka
 * cold-start).
 */
static constexpr const char *kFaasColdstart = "faas.coldstart";

/**
 * The unique identifier of the feature flag.
 */
static constexpr const char *kFeatureFlagKey = "feature_flag.key";

/**
 * The name of the service provider that performs the flag evaluation.
 */
static constexpr const char *kFeatureFlagProviderName = "feature_flag.provider_name";

/**
 * SHOULD be a semantic identifier for a value. If one is unavailable, a stringified version of the
value can be used.
 *
 * <p>Notes:
  <ul> <li>A semantic identifier, commonly referred to as a variant, provides a means
for referring to a value without including the value itself. This can
provide additional context for understanding the meaning behind a value.
For example, the variant {@code red} maybe be used for the value {@code #c05543}.</li><li>A
stringified version of the value can be used in situations where a semantic identifier is
unavailable. String representation of the value should be determined by the implementer.</li> </ul>
 */
static constexpr const char *kFeatureFlagVariant = "feature_flag.variant";

/**
 * The AWS request ID as returned in the response headers {@code x-amz-request-id} or {@code
 * x-amz-requestid}.
 */
static constexpr const char *kAwsRequestId = "aws.request_id";

/**
 * The value of the {@code AttributesToGet} request parameter.
 */
static constexpr const char *kAwsDynamodbAttributesToGet = "aws.dynamodb.attributes_to_get";

/**
 * The value of the {@code ConsistentRead} request parameter.
 */
static constexpr const char *kAwsDynamodbConsistentRead = "aws.dynamodb.consistent_read";

/**
 * The JSON-serialized value of each item in the {@code ConsumedCapacity} response field.
 */
static constexpr const char *kAwsDynamodbConsumedCapacity = "aws.dynamodb.consumed_capacity";

/**
 * The value of the {@code IndexName} request parameter.
 */
static constexpr const char *kAwsDynamodbIndexName = "aws.dynamodb.index_name";

/**
 * The JSON-serialized value of the {@code ItemCollectionMetrics} response field.
 */
static constexpr const char *kAwsDynamodbItemCollectionMetrics =
    "aws.dynamodb.item_collection_metrics";

/**
 * The value of the {@code Limit} request parameter.
 */
static constexpr const char *kAwsDynamodbLimit = "aws.dynamodb.limit";

/**
 * The value of the {@code ProjectionExpression} request parameter.
 */
static constexpr const char *kAwsDynamodbProjection = "aws.dynamodb.projection";

/**
 * The value of the {@code ProvisionedThroughput.ReadCapacityUnits} request parameter.
 */
static constexpr const char *kAwsDynamodbProvisionedReadCapacity =
    "aws.dynamodb.provisioned_read_capacity";

/**
 * The value of the {@code ProvisionedThroughput.WriteCapacityUnits} request parameter.
 */
static constexpr const char *kAwsDynamodbProvisionedWriteCapacity =
    "aws.dynamodb.provisioned_write_capacity";

/**
 * The value of the {@code Select} request parameter.
 */
static constexpr const char *kAwsDynamodbSelect = "aws.dynamodb.select";

/**
 * The keys in the {@code RequestItems} object field.
 */
static constexpr const char *kAwsDynamodbTableNames = "aws.dynamodb.table_names";

/**
 * The JSON-serialized value of each item of the {@code GlobalSecondaryIndexes} request field
 */
static constexpr const char *kAwsDynamodbGlobalSecondaryIndexes =
    "aws.dynamodb.global_secondary_indexes";

/**
 * The JSON-serialized value of each item of the {@code LocalSecondaryIndexes} request field.
 */
static constexpr const char *kAwsDynamodbLocalSecondaryIndexes =
    "aws.dynamodb.local_secondary_indexes";

/**
 * The value of the {@code ExclusiveStartTableName} request parameter.
 */
static constexpr const char *kAwsDynamodbExclusiveStartTable = "aws.dynamodb.exclusive_start_table";

/**
 * The the number of items in the {@code TableNames} response parameter.
 */
static constexpr const char *kAwsDynamodbTableCount = "aws.dynamodb.table_count";

/**
 * The value of the {@code ScanIndexForward} request parameter.
 */
static constexpr const char *kAwsDynamodbScanForward = "aws.dynamodb.scan_forward";

/**
 * The value of the {@code Count} response parameter.
 */
static constexpr const char *kAwsDynamodbCount = "aws.dynamodb.count";

/**
 * The value of the {@code ScannedCount} response parameter.
 */
static constexpr const char *kAwsDynamodbScannedCount = "aws.dynamodb.scanned_count";

/**
 * The value of the {@code Segment} request parameter.
 */
static constexpr const char *kAwsDynamodbSegment = "aws.dynamodb.segment";

/**
 * The value of the {@code TotalSegments} request parameter.
 */
static constexpr const char *kAwsDynamodbTotalSegments = "aws.dynamodb.total_segments";

/**
 * The JSON-serialized value of each item in the {@code AttributeDefinitions} request field.
 */
static constexpr const char *kAwsDynamodbAttributeDefinitions =
    "aws.dynamodb.attribute_definitions";

/**
 * The JSON-serialized value of each item in the the {@code GlobalSecondaryIndexUpdates} request
 * field.
 */
static constexpr const char *kAwsDynamodbGlobalSecondaryIndexUpdates =
    "aws.dynamodb.global_secondary_index_updates";

/**
 * The S3 bucket name the request refers to. Corresponds to the {@code --bucket} parameter of the <a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html">S3 API</a> operations.
 *
 * <p>Notes:
  <ul> <li>The {@code bucket} attribute is applicable to all S3 operations that reference a bucket,
i.e. that require the bucket name as a mandatory parameter. This applies to almost all S3 operations
except {@code list-buckets}.</li> </ul>
 */
static constexpr const char *kAwsS3Bucket = "aws.s3.bucket";

/**
 * The source object (in the form {@code bucket}/{@code key}) for the copy operation.
 *
 * <p>Notes:
  <ul> <li>The {@code copy_source} attribute applies to S3 copy operations and corresponds to the
{@code --copy-source} parameter of the <a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/copy-object.html">copy-object operation
within the S3 API</a>. This applies in particular to the following operations:</li><li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/copy-object.html">copy-object</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html">upload-part-copy</a></li>
 </ul>
 */
static constexpr const char *kAwsS3CopySource = "aws.s3.copy_source";

/**
 * The delete request container that specifies the objects to be deleted.
 *
 * <p>Notes:
  <ul> <li>The {@code delete} attribute is only applicable to the <a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-object.html">delete-object</a>
operation. The {@code delete} attribute corresponds to the {@code --delete} parameter of the <a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-objects.html">delete-objects
operation within the S3 API</a>.</li> </ul>
 */
static constexpr const char *kAwsS3Delete = "aws.s3.delete";

/**
 * The S3 object key the request refers to. Corresponds to the {@code --key} parameter of the <a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html">S3 API</a> operations.
 *
 * <p>Notes:
  <ul> <li>The {@code key} attribute is applicable to all object-related S3 operations, i.e. that
require the object key as a mandatory parameter. This applies in particular to the following
operations:</li><li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/copy-object.html">copy-object</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-object.html">delete-object</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/get-object.html">get-object</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/head-object.html">head-object</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/put-object.html">put-object</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/restore-object.html">restore-object</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/select-object-content.html">select-object-content</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/abort-multipart-upload.html">abort-multipart-upload</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/complete-multipart-upload.html">complete-multipart-upload</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/create-multipart-upload.html">create-multipart-upload</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/list-parts.html">list-parts</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html">upload-part</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html">upload-part-copy</a></li>
 </ul>
 */
static constexpr const char *kAwsS3Key = "aws.s3.key";

/**
 * The part number of the part being uploaded in a multipart-upload operation. This is a positive
integer between 1 and 10,000.
 *
 * <p>Notes:
  <ul> <li>The {@code part_number} attribute is only applicable to the <a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html">upload-part</a> and
<a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html">upload-part-copy</a>
operations. The {@code part_number} attribute corresponds to the {@code --part-number} parameter of
the <a href="https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html">upload-part
operation within the S3 API</a>.</li> </ul>
 */
static constexpr const char *kAwsS3PartNumber = "aws.s3.part_number";

/**
 * Upload ID that identifies the multipart upload.
 *
 * <p>Notes:
  <ul> <li>The {@code upload_id} attribute applies to S3 multipart-upload operations and corresponds
to the {@code --upload-id} parameter of the <a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html">S3 API</a> multipart
operations. This applies in particular to the following operations:</li><li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/abort-multipart-upload.html">abort-multipart-upload</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/complete-multipart-upload.html">complete-multipart-upload</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/list-parts.html">list-parts</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html">upload-part</a></li>
<li><a
href="https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html">upload-part-copy</a></li>
 </ul>
 */
static constexpr const char *kAwsS3UploadId = "aws.s3.upload_id";

/**
 * The GraphQL document being executed.
 *
 * <p>Notes:
  <ul> <li>The value may be sanitized to exclude sensitive information.</li> </ul>
 */
static constexpr const char *kGraphqlDocument = "graphql.document";

/**
 * The name of the operation being executed.
 */
static constexpr const char *kGraphqlOperationName = "graphql.operation.name";

/**
 * The type of the operation being executed.
 */
static constexpr const char *kGraphqlOperationType = "graphql.operation.type";

/**
 * Compressed size of the message in bytes.
 */
static constexpr const char *kMessageCompressedSize = "message.compressed_size";

/**
 * MUST be calculated as two different counters starting from {@code 1} one for sent messages and
 one for received message.
 *
 * <p>Notes:
  <ul> <li>This way we guarantee that the values will be consistent between different
 implementations.</li> </ul>
 */
static constexpr const char *kMessageId = "message.id";

/**
 * Whether this is a received or sent message.
 */
static constexpr const char *kMessageType = "message.type";

/**
 * Uncompressed size of the message in bytes.
 */
static constexpr const char *kMessageUncompressedSize = "message.uncompressed_size";

// Enum definitions
namespace FaasInvokedProviderValues
{
/** Alibaba Cloud. */
static constexpr const char *kAlibabaCloud = "alibaba_cloud";
/** Amazon Web Services. */
static constexpr const char *kAws = "aws";
/** Microsoft Azure. */
static constexpr const char *kAzure = "azure";
/** Google Cloud Platform. */
static constexpr const char *kGcp = "gcp";
/** Tencent Cloud. */
static constexpr const char *kTencentCloud = "tencent_cloud";
}  // namespace FaasInvokedProviderValues

namespace FaasTriggerValues
{
/** A response to some data source operation such as a database or filesystem read/write. */
static constexpr const char *kDatasource = "datasource";
/** To provide an answer to an inbound HTTP request. */
static constexpr const char *kHttp = "http";
/** A function is set to be executed when messages are sent to a messaging system. */
static constexpr const char *kPubsub = "pubsub";
/** A function is scheduled to be executed regularly. */
static constexpr const char *kTimer = "timer";
/** If none of the others apply. */
static constexpr const char *kOther = "other";
}  // namespace FaasTriggerValues

namespace LogIostreamValues
{
/** Logs from stdout stream. */
static constexpr const char *kStdout = "stdout";
/** Events from stderr stream. */
static constexpr const char *kStderr = "stderr";
}  // namespace LogIostreamValues

namespace IosStateValues
{
/** The app has become `active`. Associated with UIKit notification `applicationDidBecomeActive`. */
static constexpr const char *kActive = "active";
/** The app is now `inactive`. Associated with UIKit notification `applicationWillResignActive`. */
static constexpr const char *kInactive = "inactive";
/** The app is now in the background. This value is associated with UIKit notification
 * `applicationDidEnterBackground`. */
static constexpr const char *kBackground = "background";
/** The app is now in the foreground. This value is associated with UIKit notification
 * `applicationWillEnterForeground`. */
static constexpr const char *kForeground = "foreground";
/** The app is about to terminate. Associated with UIKit notification `applicationWillTerminate`. */
static constexpr const char *kTerminate = "terminate";
}  // namespace IosStateValues

namespace AndroidStateValues
{
/** Any time before Activity.onResume() or, if the app has no Activity, Context.startService() has
 * been called in the app for the first time. */
static constexpr const char *kCreated = "created";
/** Any time after Activity.onPause() or, if the app has no Activity, Context.stopService() has been
 * called when the app was in the foreground state. */
static constexpr const char *kBackground = "background";
/** Any time after Activity.onResume() or, if the app has no Activity, Context.startService() has
 * been called when the app was in either the created or background states. */
static constexpr const char *kForeground = "foreground";
}  // namespace AndroidStateValues

namespace StateValues
{
/** idle. */
static constexpr const char *kIdle = "idle";
/** used. */
static constexpr const char *kUsed = "used";
}  // namespace StateValues

namespace AspnetcoreRateLimitingResultValues
{
/** Lease was acquired. */
static constexpr const char *kAcquired = "acquired";
/** Lease request was rejected by the endpoint limiter. */
static constexpr const char *kEndpointLimiter = "endpoint_limiter";
/** Lease request was rejected by the global limiter. */
static constexpr const char *kGlobalLimiter = "global_limiter";
/** Lease request was canceled. */
static constexpr const char *kRequestCanceled = "request_canceled";
}  // namespace AspnetcoreRateLimitingResultValues

namespace SignalrConnectionStatusValues
{
/** The connection was closed normally. */
static constexpr const char *kNormalClosure = "normal_closure";
/** The connection was closed due to a timeout. */
static constexpr const char *kTimeout = "timeout";
/** The connection was closed because the app is shutting down. */
static constexpr const char *kAppShutdown = "app_shutdown";
}  // namespace SignalrConnectionStatusValues

namespace SignalrTransportValues
{
/** ServerSentEvents protocol. */
static constexpr const char *kServerSentEvents = "server_sent_events";
/** LongPolling protocol. */
static constexpr const char *kLongPolling = "long_polling";
/** WebSockets protocol. */
static constexpr const char *kWebSockets = "web_sockets";
}  // namespace SignalrTransportValues

namespace JvmMemoryTypeValues
{
/** Heap memory. */
static constexpr const char *kHeap = "heap";
/** Non-heap memory. */
static constexpr const char *kNonHeap = "non_heap";
}  // namespace JvmMemoryTypeValues

namespace SystemCpuStateValues
{
/** user. */
static constexpr const char *kUser = "user";
/** system. */
static constexpr const char *kSystem = "system";
/** nice. */
static constexpr const char *kNice = "nice";
/** idle. */
static constexpr const char *kIdle = "idle";
/** iowait. */
static constexpr const char *kIowait = "iowait";
/** interrupt. */
static constexpr const char *kInterrupt = "interrupt";
/** steal. */
static constexpr const char *kSteal = "steal";
}  // namespace SystemCpuStateValues

namespace SystemMemoryStateValues
{
/** used. */
static constexpr const char *kUsed = "used";
/** free. */
static constexpr const char *kFree = "free";
/** shared. */
static constexpr const char *kShared = "shared";
/** buffers. */
static constexpr const char *kBuffers = "buffers";
/** cached. */
static constexpr const char *kCached = "cached";
}  // namespace SystemMemoryStateValues

namespace SystemPagingDirectionValues
{
/** in. */
static constexpr const char *kIn = "in";
/** out. */
static constexpr const char *kOut = "out";
}  // namespace SystemPagingDirectionValues

namespace SystemPagingStateValues
{
/** used. */
static constexpr const char *kUsed = "used";
/** free. */
static constexpr const char *kFree = "free";
}  // namespace SystemPagingStateValues

namespace SystemPagingTypeValues
{
/** major. */
static constexpr const char *kMajor = "major";
/** minor. */
static constexpr const char *kMinor = "minor";
}  // namespace SystemPagingTypeValues

namespace SystemFilesystemStateValues
{
/** used. */
static constexpr const char *kUsed = "used";
/** free. */
static constexpr const char *kFree = "free";
/** reserved. */
static constexpr const char *kReserved = "reserved";
}  // namespace SystemFilesystemStateValues

namespace SystemFilesystemTypeValues
{
/** fat32. */
static constexpr const char *kFat32 = "fat32";
/** exfat. */
static constexpr const char *kExfat = "exfat";
/** ntfs. */
static constexpr const char *kNtfs = "ntfs";
/** refs. */
static constexpr const char *kRefs = "refs";
/** hfsplus. */
static constexpr const char *kHfsplus = "hfsplus";
/** ext4. */
static constexpr const char *kExt4 = "ext4";
}  // namespace SystemFilesystemTypeValues

namespace SystemNetworkStateValues
{
/** close. */
static constexpr const char *kClose = "close";
/** close_wait. */
static constexpr const char *kCloseWait = "close_wait";
/** closing. */
static constexpr const char *kClosing = "closing";
/** delete. */
static constexpr const char *kDelete = "delete";
/** established. */
static constexpr const char *kEstablished = "established";
/** fin_wait_1. */
static constexpr const char *kFinWait1 = "fin_wait_1";
/** fin_wait_2. */
static constexpr const char *kFinWait2 = "fin_wait_2";
/** last_ack. */
static constexpr const char *kLastAck = "last_ack";
/** listen. */
static constexpr const char *kListen = "listen";
/** syn_recv. */
static constexpr const char *kSynRecv = "syn_recv";
/** syn_sent. */
static constexpr const char *kSynSent = "syn_sent";
/** time_wait. */
static constexpr const char *kTimeWait = "time_wait";
}  // namespace SystemNetworkStateValues

namespace SystemProcessesStatusValues
{
/** running. */
static constexpr const char *kRunning = "running";
/** sleeping. */
static constexpr const char *kSleeping = "sleeping";
/** stopped. */
static constexpr const char *kStopped = "stopped";
/** defunct. */
static constexpr const char *kDefunct = "defunct";
}  // namespace SystemProcessesStatusValues

namespace DbCassandraConsistencyLevelValues
{
/** all. */
static constexpr const char *kAll = "all";
/** each_quorum. */
static constexpr const char *kEachQuorum = "each_quorum";
/** quorum. */
static constexpr const char *kQuorum = "quorum";
/** local_quorum. */
static constexpr const char *kLocalQuorum = "local_quorum";
/** one. */
static constexpr const char *kOne = "one";
/** two. */
static constexpr const char *kTwo = "two";
/** three. */
static constexpr const char *kThree = "three";
/** local_one. */
static constexpr const char *kLocalOne = "local_one";
/** any. */
static constexpr const char *kAny = "any";
/** serial. */
static constexpr const char *kSerial = "serial";
/** local_serial. */
static constexpr const char *kLocalSerial = "local_serial";
}  // namespace DbCassandraConsistencyLevelValues

namespace DbCosmosdbConnectionModeValues
{
/** Gateway (HTTP) connections mode. */
static constexpr const char *kGateway = "gateway";
/** Direct connection. */
static constexpr const char *kDirect = "direct";
}  // namespace DbCosmosdbConnectionModeValues

namespace DbCosmosdbOperationTypeValues
{
/** invalid. */
static constexpr const char *kInvalid = "Invalid";
/** create. */
static constexpr const char *kCreate = "Create";
/** patch. */
static constexpr const char *kPatch = "Patch";
/** read. */
static constexpr const char *kRead = "Read";
/** read_feed. */
static constexpr const char *kReadFeed = "ReadFeed";
/** delete. */
static constexpr const char *kDelete = "Delete";
/** replace. */
static constexpr const char *kReplace = "Replace";
/** execute. */
static constexpr const char *kExecute = "Execute";
/** query. */
static constexpr const char *kQuery = "Query";
/** head. */
static constexpr const char *kHead = "Head";
/** head_feed. */
static constexpr const char *kHeadFeed = "HeadFeed";
/** upsert. */
static constexpr const char *kUpsert = "Upsert";
/** batch. */
static constexpr const char *kBatch = "Batch";
/** query_plan. */
static constexpr const char *kQueryPlan = "QueryPlan";
/** execute_javascript. */
static constexpr const char *kExecuteJavascript = "ExecuteJavaScript";
}  // namespace DbCosmosdbOperationTypeValues

namespace DbSystemValues
{
/** Some other SQL database. Fallback only. See notes. */
static constexpr const char *kOtherSql = "other_sql";
/** Microsoft SQL Server. */
static constexpr const char *kMssql = "mssql";
/** Microsoft SQL Server Compact. */
static constexpr const char *kMssqlcompact = "mssqlcompact";
/** MySQL. */
static constexpr const char *kMysql = "mysql";
/** Oracle Database. */
static constexpr const char *kOracle = "oracle";
/** IBM Db2. */
static constexpr const char *kDb2 = "db2";
/** PostgreSQL. */
static constexpr const char *kPostgresql = "postgresql";
/** Amazon Redshift. */
static constexpr const char *kRedshift = "redshift";
/** Apache Hive. */
static constexpr const char *kHive = "hive";
/** Cloudscape. */
static constexpr const char *kCloudscape = "cloudscape";
/** HyperSQL DataBase. */
static constexpr const char *kHsqldb = "hsqldb";
/** Progress Database. */
static constexpr const char *kProgress = "progress";
/** SAP MaxDB. */
static constexpr const char *kMaxdb = "maxdb";
/** SAP HANA. */
static constexpr const char *kHanadb = "hanadb";
/** Ingres. */
static constexpr const char *kIngres = "ingres";
/** FirstSQL. */
static constexpr const char *kFirstsql = "firstsql";
/** EnterpriseDB. */
static constexpr const char *kEdb = "edb";
/** InterSystems Caché. */
static constexpr const char *kCache = "cache";
/** Adabas (Adaptable Database System). */
static constexpr const char *kAdabas = "adabas";
/** Firebird. */
static constexpr const char *kFirebird = "firebird";
/** Apache Derby. */
static constexpr const char *kDerby = "derby";
/** FileMaker. */
static constexpr const char *kFilemaker = "filemaker";
/** Informix. */
static constexpr const char *kInformix = "informix";
/** InstantDB. */
static constexpr const char *kInstantdb = "instantdb";
/** InterBase. */
static constexpr const char *kInterbase = "interbase";
/** MariaDB. */
static constexpr const char *kMariadb = "mariadb";
/** Netezza. */
static constexpr const char *kNetezza = "netezza";
/** Pervasive PSQL. */
static constexpr const char *kPervasive = "pervasive";
/** PointBase. */
static constexpr const char *kPointbase = "pointbase";
/** SQLite. */
static constexpr const char *kSqlite = "sqlite";
/** Sybase. */
static constexpr const char *kSybase = "sybase";
/** Teradata. */
static constexpr const char *kTeradata = "teradata";
/** Vertica. */
static constexpr const char *kVertica = "vertica";
/** H2. */
static constexpr const char *kH2 = "h2";
/** ColdFusion IMQ. */
static constexpr const char *kColdfusion = "coldfusion";
/** Apache Cassandra. */
static constexpr const char *kCassandra = "cassandra";
/** Apache HBase. */
static constexpr const char *kHbase = "hbase";
/** MongoDB. */
static constexpr const char *kMongodb = "mongodb";
/** Redis. */
static constexpr const char *kRedis = "redis";
/** Couchbase. */
static constexpr const char *kCouchbase = "couchbase";
/** CouchDB. */
static constexpr const char *kCouchdb = "couchdb";
/** Microsoft Azure Cosmos DB. */
static constexpr const char *kCosmosdb = "cosmosdb";
/** Amazon DynamoDB. */
static constexpr const char *kDynamodb = "dynamodb";
/** Neo4j. */
static constexpr const char *kNeo4j = "neo4j";
/** Apache Geode. */
static constexpr const char *kGeode = "geode";
/** Elasticsearch. */
static constexpr const char *kElasticsearch = "elasticsearch";
/** Memcached. */
static constexpr const char *kMemcached = "memcached";
/** CockroachDB. */
static constexpr const char *kCockroachdb = "cockroachdb";
/** OpenSearch. */
static constexpr const char *kOpensearch = "opensearch";
/** ClickHouse. */
static constexpr const char *kClickhouse = "clickhouse";
/** Cloud Spanner. */
static constexpr const char *kSpanner = "spanner";
/** Trino. */
static constexpr const char *kTrino = "trino";
}  // namespace DbSystemValues

namespace HttpFlavorValues
{
/** HTTP/1.0. */
static constexpr const char *kHttp10 = "1.0";
/** HTTP/1.1. */
static constexpr const char *kHttp11 = "1.1";
/** HTTP/2. */
static constexpr const char *kHttp20 = "2.0";
/** HTTP/3. */
static constexpr const char *kHttp30 = "3.0";
/** SPDY protocol. */
static constexpr const char *kSpdy = "SPDY";
/** QUIC protocol. */
static constexpr const char *kQuic = "QUIC";
}  // namespace HttpFlavorValues

namespace NetSockFamilyValues
{
/** IPv4 address. */
static constexpr const char *kInet = "inet";
/** IPv6 address. */
static constexpr const char *kInet6 = "inet6";
/** Unix domain socket path. */
static constexpr const char *kUnix = "unix";
}  // namespace NetSockFamilyValues

namespace NetTransportValues
{
/** ip_tcp. */
static constexpr const char *kIpTcp = "ip_tcp";
/** ip_udp. */
static constexpr const char *kIpUdp = "ip_udp";
/** Named or anonymous pipe. */
static constexpr const char *kPipe = "pipe";
/** In-process communication. */
static constexpr const char *kInproc = "inproc";
/** Something else (non IP-based). */
static constexpr const char *kOther = "other";
}  // namespace NetTransportValues

namespace DiskIoDirectionValues
{
/** read. */
static constexpr const char *kRead = "read";
/** write. */
static constexpr const char *kWrite = "write";
}  // namespace DiskIoDirectionValues

namespace ErrorTypeValues
{
/** A fallback error value to be used when the instrumentation doesn&#39;t define a custom value. */
static constexpr const char *kOther = "_OTHER";
}  // namespace ErrorTypeValues

namespace HttpRequestMethodValues
{
/** CONNECT method. */
static constexpr const char *kConnect = "CONNECT";
/** DELETE method. */
static constexpr const char *kDelete = "DELETE";
/** GET method. */
static constexpr const char *kGet = "GET";
/** HEAD method. */
static constexpr const char *kHead = "HEAD";
/** OPTIONS method. */
static constexpr const char *kOptions = "OPTIONS";
/** PATCH method. */
static constexpr const char *kPatch = "PATCH";
/** POST method. */
static constexpr const char *kPost = "POST";
/** PUT method. */
static constexpr const char *kPut = "PUT";
/** TRACE method. */
static constexpr const char *kTrace = "TRACE";
/** Any HTTP method that the instrumentation has no prior knowledge of. */
static constexpr const char *kOther = "_OTHER";
}  // namespace HttpRequestMethodValues

namespace MessagingOperationValues
{
/** One or more messages are provided for publishing to an intermediary. If a single message is
 * published, the context of the &#34;Publish&#34; span can be used as the creation context and no
 * &#34;Create&#34; span needs to be created. */
static constexpr const char *kPublish = "publish";
/** A message is created. &#34;Create&#34; spans always refer to a single message and are used to
 * provide a unique creation context for messages in batch publishing scenarios. */
static constexpr const char *kCreate = "create";
/** One or more messages are requested by a consumer. This operation refers to pull-based scenarios,
 * where consumers explicitly call methods of messaging SDKs to receive messages. */
static constexpr const char *kReceive = "receive";
/** One or more messages are passed to a consumer. This operation refers to push-based scenarios,
 * where consumer register callbacks which get called by messaging SDKs. */
static constexpr const char *kDeliver = "deliver";
}  // namespace MessagingOperationValues

namespace MessagingRocketmqConsumptionModelValues
{
/** Clustering consumption model. */
static constexpr const char *kClustering = "clustering";
/** Broadcasting consumption model. */
static constexpr const char *kBroadcasting = "broadcasting";
}  // namespace MessagingRocketmqConsumptionModelValues

namespace MessagingRocketmqMessageTypeValues
{
/** Normal message. */
static constexpr const char *kNormal = "normal";
/** FIFO message. */
static constexpr const char *kFifo = "fifo";
/** Delay message. */
static constexpr const char *kDelay = "delay";
/** Transaction message. */
static constexpr const char *kTransaction = "transaction";
}  // namespace MessagingRocketmqMessageTypeValues

namespace MessagingSystemValues
{
/** Apache ActiveMQ. */
static constexpr const char *kActivemq = "activemq";
/** Amazon Simple Queue Service (SQS). */
static constexpr const char *kAwsSqs = "aws_sqs";
/** Azure Event Grid. */
static constexpr const char *kAzureEventgrid = "azure_eventgrid";
/** Azure Event Hubs. */
static constexpr const char *kAzureEventhubs = "azure_eventhubs";
/** Azure Service Bus. */
static constexpr const char *kAzureServicebus = "azure_servicebus";
/** Google Cloud Pub/Sub. */
static constexpr const char *kGcpPubsub = "gcp_pubsub";
/** Java Message Service. */
static constexpr const char *kJms = "jms";
/** Apache Kafka. */
static constexpr const char *kKafka = "kafka";
/** RabbitMQ. */
static constexpr const char *kRabbitmq = "rabbitmq";
/** Apache RocketMQ. */
static constexpr const char *kRocketmq = "rocketmq";
}  // namespace MessagingSystemValues

namespace NetworkConnectionSubtypeValues
{
/** GPRS. */
static constexpr const char *kGprs = "gprs";
/** EDGE. */
static constexpr const char *kEdge = "edge";
/** UMTS. */
static constexpr const char *kUmts = "umts";
/** CDMA. */
static constexpr const char *kCdma = "cdma";
/** EVDO Rel. 0. */
static constexpr const char *kEvdo0 = "evdo_0";
/** EVDO Rev. A. */
static constexpr const char *kEvdoA = "evdo_a";
/** CDMA2000 1XRTT. */
static constexpr const char *kCdma20001xrtt = "cdma2000_1xrtt";
/** HSDPA. */
static constexpr const char *kHsdpa = "hsdpa";
/** HSUPA. */
static constexpr const char *kHsupa = "hsupa";
/** HSPA. */
static constexpr const char *kHspa = "hspa";
/** IDEN. */
static constexpr const char *kIden = "iden";
/** EVDO Rev. B. */
static constexpr const char *kEvdoB = "evdo_b";
/** LTE. */
static constexpr const char *kLte = "lte";
/** EHRPD. */
static constexpr const char *kEhrpd = "ehrpd";
/** HSPAP. */
static constexpr const char *kHspap = "hspap";
/** GSM. */
static constexpr const char *kGsm = "gsm";
/** TD-SCDMA. */
static constexpr const char *kTdScdma = "td_scdma";
/** IWLAN. */
static constexpr const char *kIwlan = "iwlan";
/** 5G NR (New Radio). */
static constexpr const char *kNr = "nr";
/** 5G NRNSA (New Radio Non-Standalone). */
static constexpr const char *kNrnsa = "nrnsa";
/** LTE CA. */
static constexpr const char *kLteCa = "lte_ca";
}  // namespace NetworkConnectionSubtypeValues

namespace NetworkConnectionTypeValues
{
/** wifi. */
static constexpr const char *kWifi = "wifi";
/** wired. */
static constexpr const char *kWired = "wired";
/** cell. */
static constexpr const char *kCell = "cell";
/** unavailable. */
static constexpr const char *kUnavailable = "unavailable";
/** unknown. */
static constexpr const char *kUnknown = "unknown";
}  // namespace NetworkConnectionTypeValues

namespace NetworkIoDirectionValues
{
/** transmit. */
static constexpr const char *kTransmit = "transmit";
/** receive. */
static constexpr const char *kReceive = "receive";
}  // namespace NetworkIoDirectionValues

namespace NetworkTransportValues
{
/** TCP. */
static constexpr const char *kTcp = "tcp";
/** UDP. */
static constexpr const char *kUdp = "udp";
/** Named or anonymous pipe. */
static constexpr const char *kPipe = "pipe";
/** Unix domain socket. */
static constexpr const char *kUnix = "unix";
}  // namespace NetworkTransportValues

namespace NetworkTypeValues
{
/** IPv4. */
static constexpr const char *kIpv4 = "ipv4";
/** IPv6. */
static constexpr const char *kIpv6 = "ipv6";
}  // namespace NetworkTypeValues

namespace RpcConnectRpcErrorCodeValues
{
/** cancelled. */
static constexpr const char *kCancelled = "cancelled";
/** unknown. */
static constexpr const char *kUnknown = "unknown";
/** invalid_argument. */
static constexpr const char *kInvalidArgument = "invalid_argument";
/** deadline_exceeded. */
static constexpr const char *kDeadlineExceeded = "deadline_exceeded";
/** not_found. */
static constexpr const char *kNotFound = "not_found";
/** already_exists. */
static constexpr const char *kAlreadyExists = "already_exists";
/** permission_denied. */
static constexpr const char *kPermissionDenied = "permission_denied";
/** resource_exhausted. */
static constexpr const char *kResourceExhausted = "resource_exhausted";
/** failed_precondition. */
static constexpr const char *kFailedPrecondition = "failed_precondition";
/** aborted. */
static constexpr const char *kAborted = "aborted";
/** out_of_range. */
static constexpr const char *kOutOfRange = "out_of_range";
/** unimplemented. */
static constexpr const char *kUnimplemented = "unimplemented";
/** internal. */
static constexpr const char *kInternal = "internal";
/** unavailable. */
static constexpr const char *kUnavailable = "unavailable";
/** data_loss. */
static constexpr const char *kDataLoss = "data_loss";
/** unauthenticated. */
static constexpr const char *kUnauthenticated = "unauthenticated";
}  // namespace RpcConnectRpcErrorCodeValues

namespace RpcGrpcStatusCodeValues
{
/** OK. */
static constexpr const int kOk = 0;
/** CANCELLED. */
static constexpr const int kCancelled = 1;
/** UNKNOWN. */
static constexpr const int kUnknown = 2;
/** INVALID_ARGUMENT. */
static constexpr const int kInvalidArgument = 3;
/** DEADLINE_EXCEEDED. */
static constexpr const int kDeadlineExceeded = 4;
/** NOT_FOUND. */
static constexpr const int kNotFound = 5;
/** ALREADY_EXISTS. */
static constexpr const int kAlreadyExists = 6;
/** PERMISSION_DENIED. */
static constexpr const int kPermissionDenied = 7;
/** RESOURCE_EXHAUSTED. */
static constexpr const int kResourceExhausted = 8;
/** FAILED_PRECONDITION. */
static constexpr const int kFailedPrecondition = 9;
/** ABORTED. */
static constexpr const int kAborted = 10;
/** OUT_OF_RANGE. */
static constexpr const int kOutOfRange = 11;
/** UNIMPLEMENTED. */
static constexpr const int kUnimplemented = 12;
/** INTERNAL. */
static constexpr const int kInternal = 13;
/** UNAVAILABLE. */
static constexpr const int kUnavailable = 14;
/** DATA_LOSS. */
static constexpr const int kDataLoss = 15;
/** UNAUTHENTICATED. */
static constexpr const int kUnauthenticated = 16;
}  // namespace RpcGrpcStatusCodeValues

namespace RpcSystemValues
{
/** gRPC. */
static constexpr const char *kGrpc = "grpc";
/** Java RMI. */
static constexpr const char *kJavaRmi = "java_rmi";
/** .NET WCF. */
static constexpr const char *kDotnetWcf = "dotnet_wcf";
/** Apache Dubbo. */
static constexpr const char *kApacheDubbo = "apache_dubbo";
/** Connect RPC. */
static constexpr const char *kConnectRpc = "connect_rpc";
}  // namespace RpcSystemValues

namespace TlsProtocolNameValues
{
/** ssl. */
static constexpr const char *kSsl = "ssl";
/** tls. */
static constexpr const char *kTls = "tls";
}  // namespace TlsProtocolNameValues

namespace OpentracingRefTypeValues
{
/** The parent Span depends on the child Span in some capacity. */
static constexpr const char *kChildOf = "child_of";
/** The parent Span doesn&#39;t depend in any way on the result of the child Span. */
static constexpr const char *kFollowsFrom = "follows_from";
}  // namespace OpentracingRefTypeValues

namespace OtelStatusCodeValues
{
/** The operation has been validated by an Application developer or Operator to have completed
 * successfully. */
static constexpr const char *kOk = "OK";
/** The operation contains an error. */
static constexpr const char *kError = "ERROR";
}  // namespace OtelStatusCodeValues

namespace FaasDocumentOperationValues
{
/** When a new object is created. */
static constexpr const char *kInsert = "insert";
/** When an object is modified. */
static constexpr const char *kEdit = "edit";
/** When an object is deleted. */
static constexpr const char *kDelete = "delete";
}  // namespace FaasDocumentOperationValues

namespace GraphqlOperationTypeValues
{
/** GraphQL query. */
static constexpr const char *kQuery = "query";
/** GraphQL mutation. */
static constexpr const char *kMutation = "mutation";
/** GraphQL subscription. */
static constexpr const char *kSubscription = "subscription";
}  // namespace GraphqlOperationTypeValues

namespace MessageTypeValues
{
/** sent. */
static constexpr const char *kSent = "SENT";
/** received. */
static constexpr const char *kReceived = "RECEIVED";
}  // namespace MessageTypeValues

}  // namespace SemanticConventions
}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
