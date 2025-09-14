from langgraph.checkpoint.postgres import PostgresSaver
from contextlib import contextmanager
from typing import Any, AsyncIterator, Iterator, Self, Sequence
from psycopg import Connection, OperationalError, Error as PsycopgError
from psycopg.rows import dict_row
from typing_extensions import override
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.postgres import PostgresSaver
import time
import logging


class AsyncPostgresSaver(PostgresSaver):
    def __init__(self, conn, pipe=None):
        super().__init__(conn, pipe)
        self._conn_string = None
        self.logger = logging.getLogger(__name__)

    def _create_fresh_connection(self):
        """Create a fresh database connection."""
        if not self._conn_string:
            raise ValueError("Connection string not set")

        return Connection.connect(
            self._conn_string,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,  # type: ignore
            connect_timeout=30,
            options="-c statement_timeout=300000 -c tcp_keepalives_idle=300 -c tcp_keepalives_interval=30 -c tcp_keepalives_count=3",
        )

    @classmethod
    @contextmanager
    def from_conn_string(
        cls, conn_string: str, *, pipeline: bool = False
    ) -> Iterator[Self]:
        """Create a new PostgresSaver instance with fresh connections.

        Args:
            conn_string: The Postgres connection info string.
            pipeline: whether to use Pipeline

        Returns:
            PostgresSaver: A new PostgresSaver instance.
        """
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                with Connection.connect(
                    conn_string,
                    autocommit=True,
                    prepare_threshold=0,
                    row_factory=dict_row,  # type: ignore
                    connect_timeout=30,
                    options="-c statement_timeout=300000 -c tcp_keepalives_idle=300 -c tcp_keepalives_interval=30 -c tcp_keepalives_count=3",
                ) as conn:
                    instance = cls(conn, None)
                    instance._conn_string = conn_string
                    if pipeline:
                        with conn.pipeline() as pipe:
                            instance.pipe = pipe
                            yield instance
                    else:
                        yield instance
                    return

            except (OperationalError, PsycopgError) as e:
                print(f"PostgreSQL connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("Max retries reached, raising exception")
                    raise
                else:
                    print(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2

    def _execute_with_fresh_connection(self, operation, *args, **kwargs):
        """Execute an operation with a fresh connection if needed."""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except (OperationalError, PsycopgError) as e:
                self.logger.warning(f"Operation failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise

                # Create fresh connection and retry
                try:
                    fresh_conn = self._create_fresh_connection()
                    old_conn = self.conn
                    self.conn = fresh_conn
                    if hasattr(old_conn, "close"):
                        try:
                            old_conn.close()
                        except:
                            pass
                    time.sleep(0.5 * (attempt + 1))
                except Exception as conn_e:
                    self.logger.error(f"Failed to create fresh connection: {conn_e}")
                    if attempt == max_retries - 1:
                        raise

    @override
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Asynchronously fetch a checkpoint tuple using the given configuration.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[CheckpointTuple]: The requested checkpoint tuple, or None if not found.
        """
        return self.get_tuple(config)

    @override
    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously list checkpoints that match the given criteria.

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: List checkpoints created before this configuration.
            limit: Maximum number of checkpoints to return.

        Returns:
            AsyncIterator[CheckpointTuple]: Async iterator of matching checkpoint tuples.
        """
        for tuple in self.list(config, filter=filter, before=before, limit=limit):
            yield tuple

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ):
        """Asynchronously store a checkpoint with its configuration and metadata."""
        try:
            return self._execute_with_fresh_connection(
                self.put, config, checkpoint, metadata, new_versions
            )
        except Exception as e:
            self.logger.error(f"Checkpoint put failed after all retries: {e}")
            # Return a valid config even if checkpoint fails
            return config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Asynchronously store intermediate writes linked to a checkpoint."""
        try:
            return self._execute_with_fresh_connection(
                self.put_writes, config, writes, task_id, task_path
            )
        except Exception as e:
            self.logger.error(f"Checkpoint write failed after all retries: {e}")
            # Don't raise - allow workflow to continue even if checkpoint fails
            self.logger.info("Continuing workflow execution despite checkpoint failure")

    async def adelete_thread(
        self,
        thread_id: str,
    ) -> None:
        """Delete all checkpoints and writes associated with a specific thread ID.

        Args:
            thread_id: The thread ID whose checkpoints should be deleted.
        """
        return self.delete_thread(thread_id)
